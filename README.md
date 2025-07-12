
# faiss-addr-sim-engine

<div style="border:2px solid #4CAF50; border-radius:8px; padding:8px; display:inline-block; background:#f0fdf4">
**faiss-addr-sim-engine**
</div>

High-performance, production-ready address similarity engine with built-in data generation, FAISS indexing, and system monitoring.

---

## 📄 Description

`faiss-addr-sim-engine` offers an end-to-end pipeline to generate realistic, human-like address variants and perform large-scale cosine-similarity searches using FAISS:

- **Parallelized Address Generator**  
  Applies configurable omission rates and typos to Chinese addresses, producing raw vs. human-like pairs.

- **Standardization & FAISS Indexer**  
  Chunks 300K+ unique addresses, vectorizes with `HashingVectorizer`, normalizes (L2), compresses to `.npz`, and builds CPU/GPU indices.

- **Multi-Process Similarity Search**  
  Uses `multiprocessing.Pool`, auto-fallback from GPU to CPU, and merges per-row results into a gzipped CSV.

- **System Monitoring**  
  Logs CPU%, RAM, and (optionally) GPU stats to CSV. Generates benchmark and resource-usage plots automatically.

---

## ✨ Key Features

<div style="border:1px solid #2196F3; border-radius:6px; padding:10px; background:#E3F2FD">
  <strong>Address Generator</strong>
  <ul>
    <li>Configurable omission/typo rates per component (分區, 地區, 城鎮, 道路, 屋苑名稱)</li>
    <li>Randomized floor, unit, and ordering for human realism</li>
    <li>Batch CSV output with progress logging</li>
  </ul>
</div>

<div style="border:1px solid #FF9800; border-radius:6px; padding:10px; background:#FFF3E0">
  <strong>FAISS Indexing</strong>
  <ul>
    <li>Chunks of 1,000–50,000 vectors, L2-normalized</li>
    <li>Compressed storage via `.npz`</li>
    <li>CPU &amp; GPU support with automatic fallback</li>
  </ul>
</div>

<div style="border:1px solid #4CAF50; border-radius:6px; padding:10px; background:#E8F5E9">
  <strong>Similarity Search</strong>
  <ul>
    <li>Parallel across CPU cores (or single GPU worker)</li>
    <li>Top-K cosine similarity via FAISS or dot-product fallback</li>
    <li>Temporary CSV per row, then merged &amp; gzipped</li>
  </ul>
</div>

<div style="border:1px solid #9C27B0; border-radius:6px; padding:10px; background:#F3E5F5">
  <strong>System Monitoring &amp; Benchmarking</strong>
  <ul>
    <li>Real-time logging: CPU%, RAM MB, GPU util/mem via NVIDIA-SMI</li>
    <li>Automated visualizations: per-address time/RAM deltas &amp; system usage</li>
    <li>Robust error capture &amp; resource-aware throttling</li>
  </ul>
</div>

---

## 🚀 Quickstart

1. **Clone &amp; Install**  
   ```bash
   git clone https://github.com/Jyusi/faiss-addr-sim-engine.git
   cd faiss-addr-sim-engine
   pip install -r requirements.txt

2. **Generate &amp; Addresses**
   ```bash
   python address_generator.py \
     --input input_addresses.csv \
     --output all_generated_addresses.csv \
     --batches 15

3. **Build &amp; FAISS &amp; Index**
   ```bash
   python Cosine_Similarity_v1.py --build-index

4. **Run &amp; Similarity &amp; Search**
   ```bash
   python Cosine_Similarity_v1.py --search \
     --input all_generated_addresses.csv \
     --output similarity_results.csv.gz \
     --top-k 4

5. **View logs and plots**
   <li>`address_generator.log` and `system_monitoring.csv`</li>
   <li>`benchmark_report.csv` and `benchmark_visualisation.png`</li>
   <li>`system_monitoring_visualisation.png`</li>li>

---

## ⚙️ Configuration
- OMISSION_RATES & TYPO_RATES: Tune per-component in address_generator.py.
- N_FEATURES, NGRAM_RANGE, STD_BATCH_SIZE: Adjust in Cosine_Similarity_v1.py.
- N_PARALLEL_HUMAN_ADDRESSES: Control worker count.
- MONITORING_INTERVAL_SEC: Change monitoring frequency.

---

## 🤝 Contributing
- Fork & create a feature branch.
- Write tests for new functionality.
- Keep logging robust and resource-aware.
- Submit a PR with detailed descriptions and benchmarks.
