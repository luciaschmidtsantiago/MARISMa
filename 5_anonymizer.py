import hmac
import hashlib
import re
import os
import csv
import sys
import base64
import binascii
import logging
import pandas as pd
from tqdm import tqdm

def _load_secret_key():
    """
    Load SECRET_KEY from environment or a local file.
    Accepts:
      - MARISMA_SECRET_KEY (base64, hex, or raw string)
      - MARISMA_SECRET_KEY_HEX (hex)
      - ~/.marisma_secret (contents: base64, hex, or raw string)
    Returns: bytes
    Exits with an explanatory message if no valid key is found.
    """
    env_key = os.environ.get('MARISMA_SECRET_KEY')
    env_key_hex = os.environ.get('MARISMA_SECRET_KEY_HEX')

    # If provided via env var, try base64 then hex then raw
    if env_key:
        try:
            return base64.b64decode(env_key)
        except Exception:
            try:
                return binascii.unhexlify(env_key)
            except Exception:
                return env_key.encode('utf-8')

    if env_key_hex:
        try:
            return binascii.unhexlify(env_key_hex)
        except Exception:
            sys.exit("Invalid MARISMA_SECRET_KEY_HEX (hex). Regenerate and set env var.")

    # Fallback to file in project root (preferred) then home directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    project_file = os.path.join(project_root, ".marisma_secret")
    for path in (project_file, os.path.expanduser('~/.marisma_secret')):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = f.read().strip()
            # Try base64 then hex then raw
            try:
                return base64.b64decode(data)
            except Exception:
                try:
                    return binascii.unhexlify(data)
                except Exception:
                    # data may already be bytes
                    return data if isinstance(data, (bytes, bytearray)) else data.encode('utf-8')

    raise RuntimeError("No MARISMA secret key found in environment or .marisma_secret in project or home directory")
    

# Set SECRET_KEY (bytes)
SECRET_KEY = _load_secret_key()

def get_hmac(input: str):
    return hmac.new(SECRET_KEY, input.encode(), hashlib.sha256).hexdigest()

def setup_logger(log_path):
    logger = logging.getLogger('anonymizer')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger

def anonymize_folder(input: str) -> str:
    base_id, _, suffix = input.partition('-')
    based_id_hashed = get_hmac(base_id)[0:8]
    return f'{based_id_hashed}-{suffix}' if suffix else based_id_hashed

def anonymize_metadata_old(input: str) -> str:
    result = re.sub(
        r'##\$PATH= <(.+)>',
        lambda x: f'##$PATH= <ANONYMIZED_{get_hmac(x[1])}>',
        input,
    )
    result = re.sub(
        r'##\$SMPNAM= <(.+)>',
        lambda x: f'##$SMPNAM= <ANONYMIZED_{get_hmac(x[1])}>',
        result,
    )
    return result

def anonymize_metadata(filepath, orig_id, anon_id, mode):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if mode == 'PATH':
        # Only replace the PATH field
        content = re.sub(rf'##\$PATH= <{re.escape(orig_id)}>', f'##$PATH= <ANONYMIZED_{anon_id}>', content)
    elif mode == 'SMPNAM':
        # Only replace the SMPNAM field
        content = re.sub(rf'##\$SMPNAM= <{re.escape(orig_id)}>', f'##$SMPNAM= <ANONYMIZED_{anon_id}>', content)
    # Replace any other direct occurrence (optional, for extra safety)
    content = content.replace(orig_id, anon_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)



def main():
    base_dir = '/MARISMa' # CHANGE TO OWN BASE DIRECTORY
    mapping_csv_path = os.path.join(base_dir, 'anonymization_mapping.csv')
    amr_labels_path = os.path.join(base_dir, 'AMR_labels.csv')
    log_path = os.path.join(base_dir, 'logs', 'anonymizer.log')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(log_path)
    mapping = []

    # Traverse the database structure
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path) or not year.isdigit():
            continue
        for genus in tqdm(os.listdir(year_path), desc=f"Processing genus in {year_path}"):
            genus_path = os.path.join(year_path, genus)
            if not os.path.isdir(genus_path):
                continue
            for species in os.listdir(genus_path):
                species_path = os.path.join(genus_path, species)
                if not os.path.isdir(species_path):
                    continue
                for identifier in os.listdir(species_path):
                    id_path = os.path.join(species_path, identifier)
                    if not os.path.isdir(id_path):
                        continue
                    anon_id = anonymize_folder(identifier)
                    mapping.append((identifier, anon_id))
                    anon_id_path = os.path.join(species_path, anon_id)
                    if not os.path.exists(anon_id_path):
                        os.rename(id_path, anon_id_path)
                        logger.info(f'Renamed {id_path.split("bacteria_id")[1]} -> {anon_id_path.split("bacteria_id")[1]}')
                    else:
                        logger.warning(f'Target folder already exists: {anon_id_path}')
                    # Now process metadata files inside the identifier folder
                    # Recursively search for metadata files in anon_id_path
                    for root, _, files in os.walk(anon_id_path):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            if fname in ['acqu', 'acqus']:
                                try:
                                    anonymize_metadata(fpath, identifier, anon_id, mode='PATH')
                                    # logger.info(f'Anonymized PATH in {fpath}')
                                except Exception as e:
                                    logger.error(f'Error anonymizing {fpath}: {e}')
                            elif fname in ['proc', 'procs']:
                                try:
                                    anonymize_metadata(fpath, identifier, anon_id, mode='SMPNAM')
                                    # logger.info(f'Anonymized SMPNAM in {fpath}')
                                except Exception as e:
                                    logger.error(f'Error anonymizing {fpath}: {e}')

    # Write mapping CSV
    with open(mapping_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original_identifier', 'anonymized_identifier'])
        for orig_id, anon_id in mapping:
            writer.writerow([orig_id, anon_id])
    logger.info(f'Wrote anonymization mapping to {mapping_csv_path}')

    # Anonymize AMR_labels.csv
    if os.path.exists(amr_labels_path):
        df_amr = pd.read_csv(amr_labels_path, sep=';', encoding='utf-8-sig', dtype=str)

        # Read mapping from CSV
        id_map = {}
        if os.path.exists(mapping_csv_path):
            with open(mapping_csv_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    id_map[row['original_identifier']] = row['anonymized_identifier']

        if 'Identifier' in df_amr.columns:
            df_amr['Identifier'] = df_amr['Identifier'].map(lambda x: id_map.get(x, x))
        if 'Path' in df_amr.columns:
            def replace_path_id(path):
                for orig_id, anon_id in id_map.items():
                    if orig_id in str(path):
                        return str(path).replace(orig_id, anon_id)
                return path
            df_amr['Path'] = df_amr['Path'].map(replace_path_id)
        df_amr.to_csv(amr_labels_path, sep=';', encoding='utf-8-sig', index=False)
        logger.info(f'Anonymized AMR_labels.csv at {amr_labels_path}')

if __name__ == '__main__':
    main()
