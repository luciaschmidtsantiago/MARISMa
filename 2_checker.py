## Code by Inés López-Mareca and Lucía Schmidt-Santiago. Copyrght.

import os
import pandas as pd
import logging
from tqdm import tqdm
import shutil
import hashlib
import argparse

def setup_logging(log_dir):
    """Sets up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    action_log = os.path.join(log_dir, 'checker.log')
    logging.basicConfig(filename=action_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    return action_log

def delete_empty_dirs_recursive(base_dir, path):
    """Recursively deletes path and its empty parents, logs the action."""
    if not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path)
        logging.info(f"Deleted directory (recursively): {path}")
    except Exception as e:
        logging.error(f"Error deleting directory {path}: {e}")
        return
    # Now check parent
    parent = os.path.dirname(path)
    base_stop = base_dir
    while parent != base_stop and os.path.isdir(parent) and not os.listdir(parent):
        # if not os.listdir(parent) or os.listdir(parent) == ['.DS_Store']:
        try:
            shutil.rmtree(parent)
            logging.info(f"Deleted empty parent directory: {parent}")
        except Exception as e:
            logging.error(f"Error deleting parent directory {parent}: {e}")
            break
        parent = os.path.dirname(parent)

def process_fid(fid_path, df, base_dir):
    """Processes a fid file: checks for all-zero content and duplicates."""
    try:
        with open(fid_path, 'rb') as f:
            content = f.read()
        hex_content = content.hex()
        if set(hex_content.strip()) in [{'0'}, set()]:
            logging.info(f"❌ Deleting {fid_path} due to all-zero content.")
            content_todelete = os.path.dirname(os.path.dirname(fid_path))
            delete_empty_dirs_recursive(base_dir, content_todelete)
        else:
            hash_value = hashlib.sha256(content).hexdigest()
            if hash_value in df['hash'].values:
                existing_path = df[df['hash'] == hash_value]['path'].values[0]
                logging.info(f"❌ {fid_path} and {existing_path} have the same hash. Deleting both.")
                for _ in [fid_path, existing_path]:
                    content_todelete = os.path.dirname(os.path.dirname(fid_path))
                    delete_empty_dirs_recursive(base_dir, content_todelete)
                df = df[df['hash'] != hash_value]
            else:
                new_row = pd.DataFrame({'hash': [hash_value], 'path': [fid_path]})
                df = pd.concat([df, new_row], ignore_index=True)
    except Exception as e:
        logging.error(f"Error at {fid_path}: {str(e)}\n")
    return df

def count_spectra(base_dir, years):
    """Counts spectra and processes fid files."""
    df_hash = pd.DataFrame(columns=['hash', 'path']) # Initialize empty DataFrame for hashes

    log_directory = os.path.join(os.path.join(base_dir, 'logs'))
    os.makedirs(log_directory, exist_ok=True)
    setup_logging(log_directory)

    for year in years:
        print(f"Processing year: {year}")

        valid_dir = ''
        year_dir = os.path.join(base_dir, str(year))
        for root, dirs, files in tqdm(os.walk(year_dir), desc=f"Year {year}"):
            # Only count files in the deepest directories (target_position)
            if 'fid' in files:  # Move up to the directory containing 'fid'
                valid_dir = os.sep.join(root.split(os.sep)[:11])
                # Hash checking
                fid_path = os.path.join(root, 'fid')
                df_hash = process_fid(fid_path, df_hash, base_dir)

            if not dirs:
                this_dir = os.sep.join(root.split(os.sep)[:11])
                if this_dir != valid_dir:
                    logging.info(f"Unfound fid for: {this_dir}")
                    delete_empty_dirs_recursive(base_dir, this_dir)

    return df_hash

def main(base_dest_dir, years):

    # Check spectra for empty and duplicate fids
    df_hash = count_spectra(base_dest_dir, years)

    # DataFrame for hashes
    hashes_csv = 'fid_hashes.csv'
    hash_path = os.path.join(os.path.join(base_dest_dir, 'logs' ) , hashes_csv)
    df_hash.to_csv(hash_path, index=False)
    print(f"Hash data saved to {hash_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MARISMa checker script.")
    parser.add_argument("--base_dest_dir", type=str, default='/MARISMa', help="Base destination directory for output")
    parser.add_argument("years", nargs='+', type=int, help="Year(s) to process (e.g. 2019 2020)")
    args = parser.parse_args()    

    print(f"Base destination directory: {args.base_dest_dir}")
    print(f"Years to process: {args.years}")
    
    main(args.base_dest_dir, args.years)

    # main('/Volumes/data_ml4ds/bacteria_id/MARISMa', [2018])

