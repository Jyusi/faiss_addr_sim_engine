import os
import glob
import logging
import time
import psutil
import gzip
import shutil
import heapq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
# from memory_profiler import profile # This is typically for single-run analysis, not continuous monitoring
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm
import subprocess # For running nvidia-smi and htop commands
import csv
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
    try:
        _ = faiss.StandardGpuResources()
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = GPU_AVAILABLE = False

# Optional: track GPU memory using pynvml
def gpu_mem():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2  # in MB
    except Exception:
        return -1

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========== Config ==========
CHUNK_SIZE = 1_000 # Unused in current version, but kept for context
STD_BATCH_SIZE = 1_000
N_FEATURES = 50 # Accuracy and 
NGRAM_RANGE = (2, 2)
ENCODING = 'utf-8-sig'
TOP_K = 4
BENCHMARK_FILE = "benchmark_report.csv"
TEMP_DIR = "temp_std_batches"
OUTPUT_CSV = "similarity_output/final_similarity_output.csv"
OUTPUT_GZIP = OUTPUT_CSV + ".gz"
MONITORING_INTERVAL_SEC = 3 # How often to record system stats
MONITORING_LOG_FILE = "system_monitoring.csv"

# Global variables for worker processes (Option A caching)
_cached_std_data = {}
_cached_faiss_indices = {}
_gpu_res = None
_vectorizer = None # Each worker will have its own vectorizer

# ========== Core Functions ==========

def get_standardized_vector_batches(file_path, vectorizer):
    os.makedirs(TEMP_DIR, exist_ok=True)
    real_address_df = pd.read_csv(file_path, usecols=['Real_Address'], encoding=ENCODING, dtype='str').fillna('')
    standardized_texts = real_address_df['Real_Address'].drop_duplicates().tolist()
    sparse_matrix = vectorizer.transform(standardized_texts)
    vec_standardized = normalize(sparse_matrix, norm='l2', axis=1).toarray().astype(np.float32)

    batch_paths = []
    for i in range(0, len(vec_standardized), STD_BATCH_SIZE):
        batch_texts = standardized_texts[i:i+STD_BATCH_SIZE]
        batch_vecs = vec_standardized[i:i+STD_BATCH_SIZE]
        path = os.path.join(TEMP_DIR, f"std_batch_{i//STD_BATCH_SIZE:03}.npz")
        np.savez_compressed(path, texts=batch_texts, vecs=batch_vecs)
        batch_paths.append(path)
    return batch_paths

# In init_worker function
def init_worker(batch_paths, vectorizer_config):
    """
    Initializes each worker process by loading standardized data and
    creating FAISS indices (if GPU available).
    This function runs once per process when the Pool is created.
    """
    # Declare all global variables at the very beginning of the function
    # that this function might modify or read from the module level.
    global _cached_std_data, _cached_faiss_indices, _gpu_res, _vectorizer, GPU_AVAILABLE

    process_name = current_process().name
    logger.info(f"Worker {process_name} initializing...")

    # Initialize vectorizer for this worker process
    _vectorizer = HashingVectorizer(**vectorizer_config)

    # Initialize GPU resources if available
    # The 'global GPU_AVAILABLE' declaration above now correctly allows this read.
    if FAISS_AVAILABLE and GPU_AVAILABLE:
        try:
            _gpu_res = faiss.StandardGpuResources()
            logger.info(f"Worker {process_name} initialized GPU resources.")
        except Exception as e:
            logger.warning(f"Worker {process_name} failed to initialize GPU resources: {e}. Disabling GPU for this worker.")
            # This assignment is now valid because GPU_AVAILABLE was declared global above.
            GPU_AVAILABLE = False # This changes the module-level GPU_AVAILABLE for THIS process
    else:
        # If FAISS is not available at all, or GPU_AVAILABLE was already False
        # from the main process due to initial checks, ensure this worker also
        # considers GPU as not available for its operations.
        GPU_AVAILABLE = False


    # Load standardized data and create FAISS indices for each batch
    # We now check the worker's current GPU_AVAILABLE status (which might have been turned off above)
    for path in batch_paths:
        data = np.load(path, allow_pickle=True)
        std_texts = data['texts']
        std_vecs = data['vecs']
        _cached_std_data[path] = {'texts': std_texts, 'vecs': std_vecs}

        # Check GPU_AVAILABLE specific to this worker, and ensure _gpu_res was successfully set up
        if FAISS_AVAILABLE and GPU_AVAILABLE and _gpu_res: # Use current GPU_AVAILABLE and _gpu_res
            try:
                index_flat = faiss.IndexFlatIP(std_vecs.shape[1])
                gpu_index = faiss.index_cpu_to_gpu(_gpu_res, 0, index_flat)
                gpu_index.add(std_vecs)
                _cached_faiss_indices[path] = gpu_index
            except Exception as e:
                logger.warning(f"Worker {process_name} failed to create GPU index for batch {path}: {e}. Falling back to CPU for this batch.")
                _cached_faiss_indices[path] = None # Mark as no GPU index for this batch
        else:
            # If GPU is not available for this worker, or index creation failed for this batch,
            # we explicitly set it to None to ensure CPU path is taken later.
            _cached_faiss_indices[path] = None

    logger.info(f"Worker {process_name} finished initialization.")

def compute_similarity_topk_optimized(human_vec, human_text, top_k):
    """
    Computes top-k similarities using pre-loaded/pre-indexed standardized data.
    """
    global _cached_std_data, _cached_faiss_indices, GPU_AVAILABLE # Access global variable

    all_scores = []
    # Iterate through the cached standardized data batches
    for path in sorted(_cached_std_data.keys()): # Ensure consistent order
        data = _cached_std_data[path]
        std_texts = data['texts']
        std_vecs = data['vecs']

        if GPU_AVAILABLE and _cached_faiss_indices.get(path) is not None:
            # Use the pre-created GPU index for this batch
            try:
                gpu_index = _cached_faiss_indices[path]
                scores, indices = gpu_index.search(human_vec.reshape(1, -1), top_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1:
                        all_scores.append((score, std_texts[idx]))
            except Exception as e:
                logger.warning(f"⚠ Worker GPU search failed for batch {path}, falling back to CPU: {e}")
                scores = np.dot(std_vecs, human_vec.T).flatten()
                for i, score in enumerate(scores):
                    all_scores.append((score, std_texts[i]))
        else:
            # Fallback to CPU calculation
            scores = np.dot(std_vecs, human_vec.T).flatten()
            for i, score in enumerate(scores):
                all_scores.append((score, std_texts[i]))

    top_matches = heapq.nlargest(top_k, all_scores, key=lambda x: x[0])
    return [{'Human_Like_Address': human_text, 'Standardized_Address': addr, 'similarity_score': score} for score, addr in top_matches]

def process_one(args):
    """
    Processes a single human address. This function runs in a worker process.
    """
    idx, line = args # Vectorizer and batch_paths are now global in the worker process
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem = process.memory_info().rss / (1024 ** 2)
    start_gpu_mem = gpu_mem()

    fields = line.strip().split(',')
    if len(fields) < 3:
        return None
    human_text = fields[2]
    
    # Use the vectorizer initialized globally in this worker process
    human_vec = _vectorizer.transform([human_text])
    human_vec = normalize(human_vec, norm='l2').toarray().astype(np.float32)[0]

    top_results = compute_similarity_topk_optimized(human_vec, human_text, TOP_K)
    df = pd.DataFrame(top_results)

    temp_output = f"temp_output_row_{idx}.csv"
    df.to_csv(temp_output, index=False, encoding=ENCODING)

    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 ** 2)
    end_gpu_mem = gpu_mem()
    gpu_delta = end_gpu_mem - start_gpu_mem if end_gpu_mem != -1 and start_gpu_mem != -1 else 0

    return (idx + 1, end_time - start_time, end_mem - start_mem, end_mem, temp_output, gpu_delta)

def merge_temp_results(temp_files, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding=ENCODING) as f_out:
        # Write header from the first file
        if temp_files:
            with open(sorted(temp_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[0], 'r', encoding=ENCODING) as f_in_header:
                f_out.write(f_in_header.readline())

        for i, temp_file in enumerate(sorted(temp_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))):
            with open(temp_file, 'r', encoding=ENCODING) as f_in:
                if i != 0:
                    next(f_in)  # skip header for subsequent files
                shutil.copyfileobj(f_in, f_out)
            os.remove(temp_file)

def compress_final_output(output_file, compressed_file):
    with open(output_file, 'rb') as f_in:
        with gzip.open(compressed_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def visualize_benchmark(benchmark_csv):
    df = pd.read_csv(benchmark_csv)
    if not {'human_id', 'time_sec', 'ram_delta_MB'}.issubset(df.columns):
        logger.error("❌ Benchmark CSV missing required columns.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df['human_id'], df['time_sec'], label='Time (s)', marker='o')
    plt.plot(df['human_id'], df['ram_delta_MB'], label='RAM Δ (MB)', marker='x')
    if 'gpu_delta_MB' in df.columns:
        plt.plot(df['human_id'], df['gpu_delta_MB'], label='GPU Δ (MB)', marker='^')
    plt.xlabel('Human Address Index')
    plt.ylabel('Value')
    plt.title('Benchmark per Human Address')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("benchmark_visualization.png")
    plt.close() # Close plot to prevent display issues in non-interactive environments

def visualize_monitoring_data(monitoring_csv):
    try:
        df = pd.read_csv(monitoring_csv)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

        # CPU Usage
        axs[0].plot(df['timestamp'], df['cpu_percent'], label='CPU Usage (%)', color='blue')
        axs[0].set_ylabel('CPU Usage (%)')
        axs[0].set_title('System Resource Usage Over Time')
        axs[0].grid(True)
        axs[0].legend()

        # RAM Usage
        axs[1].plot(df['timestamp'], df['ram_used_mb'], label='RAM Used (MB)', color='green')
        axs[1].set_ylabel('RAM Used (MB)')
        axs[1].grid(True)
        axs[1].legend()

        # GPU Usage (if available)
        if 'gpu_util_percent' in df.columns:
            axs[2].plot(df['timestamp'], df['gpu_util_percent'], label='GPU Utilization (%)', color='red')
            axs[2].set_ylabel('GPU Utilization (%)')
            axs[2].grid(True)
            axs[2].legend()
        if 'gpu_mem_used_mb' in df.columns:
            axs[2].plot(df['timestamp'], df['gpu_mem_used_mb'], label='GPU Memory Used (MB)', color='purple', linestyle='--')
            axs[2].set_ylabel('GPU Utilization / Mem Used')
            axs[2].grid(True)
            axs[2].legend()
        
        axs[2].set_xlabel('Time')
        fig.autofmt_xdate() # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig("system_monitoring_visualization.png")
        plt.close()
        logger.info(f"📊 System monitoring visualization saved to system_monitoring_visualization.png")

    except FileNotFoundError:
        logger.warning(f"Monitoring log file not found: {monitoring_csv}. Skipping visualization.")
    except Exception as e:
        logger.error(f"Error visualizing monitoring data: {e}")

def monitor_system(log_file, interval, stop_event):
    """
    Collects system-wide CPU, RAM, and GPU stats periodically and logs them.
    This runs in a separate process/thread.
    """
    logger.info(f"Starting system monitoring to {log_file} every {interval} seconds.")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['timestamp', 'cpu_percent', 'ram_used_mb', 'ram_total_mb']
        if GPU_AVAILABLE:
            header.extend(['gpu_util_percent', 'gpu_mem_used_mb'])
        writer.writerow(header)

        while not stop_event.is_set():
            timestamp = datetime.now().isoformat()
            cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking call
            mem = psutil.virtual_memory()
            ram_used_mb = mem.used / (1024**2)
            ram_total_mb = mem.total / (1024**2)

            row = [timestamp, cpu_percent, ram_used_mb, ram_total_mb]

            if GPU_AVAILABLE:
                gpu_util_percent = -1
                gpu_mem_used_mb_val = -1
                try:
                    # Using subprocess to call nvidia-smi for overall GPU utilization
                    # Note: pynvml is better for memory, but nvidia-smi gives utilization
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True
                    )
                    output_lines = result.stdout.strip().split('\n')
                    if output_lines:
                        # Assuming single GPU (index 0) or taking first GPU's data
                        gpu_stats = output_lines[0].split(',')
                        gpu_util_percent = float(gpu_stats[0].strip())
                        gpu_mem_used_mb_val = float(gpu_stats[1].strip())
                except Exception as e:
                    logger.warning(f"Could not get detailed GPU stats with nvidia-smi: {e}")
                    # Keep -1 if error
                row.extend([gpu_util_percent, gpu_mem_used_mb_val])
            
            writer.writerow(row)
            f.flush() # Ensure data is written to disk
            time.sleep(interval)
    logger.info("System monitoring stopped.")

def main():
    file_path = 'all_generated_addresses.csv'
    if os.path.exists(BENCHMARK_FILE):
        os.remove(BENCHMARK_FILE)
    if os.path.exists(MONITORING_LOG_FILE):
        os.remove(MONITORING_LOG_FILE)

    vectorizer_config = {
        'analyzer': 'char',
        'ngram_range': NGRAM_RANGE,
        'n_features': N_FEATURES,
        'alternate_sign': False
    }
    
    # Create one vectorizer for precomputing batches (this is in main process)
    initial_vectorizer = HashingVectorizer(**vectorizer_config)
    logger.info("📌 Precomputing standardized address batches...")
    batch_paths = get_standardized_vector_batches(file_path, initial_vectorizer)
    del initial_vectorizer # Free up if not needed in main thread anymore

    with open(file_path, encoding=ENCODING) as f:
        lines = list(f.readlines())[1:] # Skip header

    # Set the number of parallel human address processing
    # Adjust this number based on your system's capabilities and desired CPU/RAM usage.
    # If GPU is critical and VRAM is limited, you might need N_PARALLEL_HUMAN_ADDRESSES = 1.
    # Otherwise, cpu_count() // 2 or 4-8 is a good starting point.
    N_PARALLEL_HUMAN_ADDRESSES = max(1, cpu_count() // 4)
    if GPU_AVAILABLE and len(batch_paths) * N_FEATURES * 4 / (1024**3) > 0.5: # Heuristic: if std_vecs might be large
        logger.warning("Large standardized dataset detected and GPU available. Consider limiting N_PARALLEL_HUMAN_ADDRESSES to 1 if VRAM is a concern.")
        N_PARALLEL_HUMAN_ADDRESSES = 1 # Uncomment to strictly limit to 1 GPU process/////////////////////////////////////////////////////////////////////////////////////////////

    logger.info(f"🚀 Processing with {N_PARALLEL_HUMAN_ADDRESSES} parallel processes...")

    # Setup for system monitoring
    from multiprocessing import Event, Process
    stop_monitoring_event = Event()
    monitor_process = Process(target=monitor_system, args=(MONITORING_LOG_FILE, MONITORING_INTERVAL_SEC, stop_monitoring_event))
    monitor_process.start()

    temp_files = []
    try:
        # Pass batch_paths and vectorizer_config to the initializer of the pool
        # Each worker will receive these arguments and use them to set up its own environment
        with Pool(N_PARALLEL_HUMAN_ADDRESSES, initializer=init_worker, initargs=(batch_paths, vectorizer_config)) as pool:
            args = [(i, line) for i, line in enumerate(lines)]
            for res in tqdm(pool.imap_unordered(process_one, args), total=len(args), desc="🔍 Computing Similarities"):
                if res:
                    human_id, time_sec, ram_delta, ram_now, temp_file, gpu_delta = res
                    temp_files.append(temp_file)
                    with open(BENCHMARK_FILE, 'a', encoding=ENCODING) as f:
                        if os.stat(BENCHMARK_FILE).st_size == 0:
                            f.write("human_id,time_sec,ram_delta_MB,ram_now_MB,gpu_delta_MB\n")
                        f.write(f"{human_id},{time_sec:.2f},{ram_delta:.2f},{ram_now:.2f},{gpu_delta:.2f}\n")
    finally:
        # Ensure monitoring stops even if an error occurs in the main pool
        stop_monitoring_event.set()
        monitor_process.join() # Wait for the monitoring process to finish

    merge_temp_results(temp_files, OUTPUT_CSV)
    compress_final_output(OUTPUT_CSV, OUTPUT_GZIP)

    logger.info("🧹 Cleaning up temporary standardized batches...")
    shutil.rmtree(TEMP_DIR)

    logger.info("📊 Visualizing benchmark...")
    visualize_benchmark(BENCHMARK_FILE)
    logger.info("📊 Visualizing system monitoring data...")
    visualize_monitoring_data(MONITORING_LOG_FILE)

    logger.info("✅ All done!")

if __name__ == '__main__':
    main()