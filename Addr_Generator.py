# Add to the top of address_generator.py
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='address_generator.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# address_generator.py
import pandas as pd
import numpy as np
import random
import io
from faker import Faker
from pandarallel import pandarallel

fake = Faker('zh_TW')

# --- Configuration ---
OMISSION_RATES = {
    '分區': 0.85,
    '地區': 0.2,
    '城鎮': 0.3,
    '道路': 0.8,
    '屋苑名稱': 0.05
}
TYPO_RATES = {
    '地區': 0.1,
    '分區': 0.1,
    '城鎮': 0.2,
    '道路': 0.2,
    '屋苑名稱': 0.1
}
FLOORS = [i for i in range(1, 42)] + ['G', 'G']
FLATS = [i for i in range(1, 20)]
UNITS = ['室', '室', 'Flat', 'Rm']
LEVELS = ['樓', '樓', '層', 'F']


def load_addresses_from_csv(csv_string):
    df = pd.read_csv(io.StringIO(csv_string))
    return df.fillna('')


def introduce_typo_chinese(text, typo_rate=0.1):
    if not isinstance(text, str) or not text:
        return text
    if random.random() >= typo_rate:
        return text

    typo_type = random.choice(['delete', 'insert', 'swap', 'replace'])
    if len(text) == 1:
        typo_type = random.choice(['insert', 'replace'])

    if typo_type == 'delete':
        idx = random.randint(0, len(text) - 1)
        return text[:idx] + text[idx+1:]
    elif typo_type == 'insert':
        idx = random.randint(0, len(text))
        insert_char = fake.word()
        return text[:idx] + insert_char + text[idx:]
    elif typo_type == 'swap' and len(text) > 1:
        idx1 = random.randint(0, len(text) - 2)
        idx2 = idx1 + 1
        char_list = list(text)
        char_list[idx1], char_list[idx2] = char_list[idx2], char_list[idx1]
        return "".join(char_list)
    elif typo_type == 'replace':
        char_list = list(text)
        num_replace = random.choice([1, 2]) if len(char_list) > 1 else 1
        idxs = random.sample(range(len(char_list)), num_replace)
        for idx in idxs:
            char_list[idx] = chr(random.randint(0x4e00, 0x9fff))
        return "".join(char_list)
    return text


def generate_human_like_address(row, omission_rates, typo_rates):
    floor = random.choice(FLOORS)
    flat = random.choice(FLATS)
    unit = random.choice(UNITS)
    level = random.choice(LEVELS)

    floor_rand = str(floor) if random.random() > 0.1 else ''
    flat_rand = str(flat) if random.random() > 0.1 else ''
    unit_rand = unit if random.random() > 0.1 else ''
    level_rand = level if random.random() > 0.1 else ''

    unit_flat = f"{flat_rand}{unit_rand}" if all('\u4e00' <= ch <= '\u9fff' for ch in unit_rand) else f"{unit_rand}{flat_rand}"
    unit_floor = f"{floor_rand}{level_rand}" if all('\u4e00' <= ch <= '\u9fff' for ch in level_rand) else f"{level_rand}{floor_rand}"

    address_components = {
        '地區': row['地區'],
        '分區': row['分區'],
        '城鎮': row['城鎮'],
        '道路': row['道路'],
        '屋苑名稱': row['屋苑名稱'],
        '樓層': unit_floor,
        '單位號碼': unit_flat
    }

    included_components = []
    original_components = [v for v in address_components.values() if v]

    for key, value in address_components.items():
        if value and random.random() > omission_rates.get(key, 0):
            typo_value = introduce_typo_chinese(value, typo_rates.get(key, 0))
            if typo_value:
                included_components.append(typo_value)

    if not included_components and row['屋苑名稱']:
        typo_value = introduce_typo_chinese(row['屋苑名稱'], typo_rates.get('屋苑名稱', 0))
        if typo_value:
            included_components.append(typo_value)

    if random.random() < 0.3:
        random.shuffle(included_components)

    return pd.Series({
        '屋苑名稱': row['屋苑名稱'],
        'Real_Address': ''.join(original_components),
        'Human_Like_Address': ''.join(included_components)
    })


def generate_addresses(df, output_file, n=1, batch_size=20000):
    total_rows = len(df)
    total_needed = n * total_rows
    first_write = True

    for start in range(0, total_needed, batch_size):
        try:
            current_batch_size = min(batch_size, total_needed - start)
            batch_indices = np.random.choice(df.index, size=current_batch_size, replace=True)
            batch_df = df.iloc[batch_indices].reset_index(drop=True)

            result_df = batch_df.parallel_apply(
                lambda row: generate_human_like_address(row, OMISSION_RATES, TYPO_RATES), axis=1
            )

            mode = 'w' if first_write else 'a'
            result_df.to_csv(output_file, encoding='utf-8-sig', index=False, header=first_write, mode=mode)
            first_write = False

            logging.info(f"Processed {start + current_batch_size} / {total_needed} rows")
        except Exception as e:
            logging.error(f"Error in batch starting at {start}: {e}", exc_info=True)
        
        print(f"Processed {start + current_batch_size} / {total_needed} rows", end='\r')


def main(csv_data, output_file='all_generated_addresses.csv'):
    try:
        logging.info("Initializing parallel processing")
        pandarallel.initialize(progress_bar=False, nb_workers=8)

        logging.info("Loading CSV data")
        df = load_addresses_from_csv(csv_data)

        logging.info("Starting address generation")
        generate_addresses(df, output_file)

        logging.info("Address generation completed successfully")
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)




# Example usage:
if __name__ == "__main__":
    with open("input_addresses.csv", "r", encoding="utf-8") as f:
        csv_data = f.read()
    main(csv_data)