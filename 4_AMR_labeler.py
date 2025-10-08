## Code by Inés López-Mareca and Lucía Schmidt-Santiago. Copyrght.

import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import re


def setup_logging(log_dir, name):
    """Sets up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    action_log = os.path.join(log_dir, name)
    logging.basicConfig(filename=action_log, level=logging.INFO, format='%(asctime)s - %(message)s')
    return action_log

def match_and_label_amr(raw_labels, labels, base_dir):
    """
    Match AMR labels from raw_labels CSV to valid spectra in base_dir and save to labels CSV.
    Args:
        raw_labels (str): Path to the raw AMR labels CSV file.
        labels (str): Path to save the matched AMR labels CSV file.
        base_dir (str): Base directory containing the stats/valid_spectra.csv file.
    """

    df_amr = pd.read_csv(raw_labels, sep=';', encoding='utf-8-sig', dtype={'Identifier': str})
    df_amr.rename(columns={df_amr.columns[0]: 'Identifier'}, inplace=True)
    df_amr.replace('-', np.nan, inplace=True)

    valid_spectra = os.path.join(base_dir, 'stats', 'valid_spectra.csv')
    df_ids = pd.read_csv(valid_spectra, dtype={'Identifier': str})

    # Build new DataFrame for matches only
    matched_rows = []
    for i, row in tqdm(df_amr.iterrows(), total=len(df_amr), desc="Matching identifiers"):
        identifier = str(row['Identifier'])
        match = df_ids[
            (df_ids['identifier'].astype(str) == identifier) |
            (df_ids['identifier'].astype(str).str.contains(re.escape(identifier), na=False, regex=True))
        ]
        if not match.empty:
            microorganism_id = str(row['Microorganism']).strip().lower()
            for idx, match_row in match.iterrows():
                genus_match = str(match_row['genus']).strip().lower()
                species_match = str(match_row['species']).strip().lower()
                if genus_match and species_match in microorganism_id:
                    for col in row.index:
                        match_row[col] = row[col]
                    matched_rows.append(match_row)
                else:
                    logging.info(f"Species mismatch for Identifier {identifier}: {genus_match} {species_match} different from {microorganism_id}")
        else:
            logging.info(f"No valid genus/species match for Identifier {identifier} in {microorganism_id}")

    # Save only matched rows to new CSV
    df_matched = pd.DataFrame(matched_rows)
    df_matched.to_csv(labels, sep=';', encoding='utf-8-sig', index=False)
    print(f"Saved {len(df_matched)} matched rows to {labels}")

def get_amr_statistics(labels):
    """
    Generate statistics from the AMR labels CSV file and save to amr_stats.json.
    Args:
        labels (str): Path to the AMR labels CSV file.
    """

    # Load the AMR labels file
    try:
        df = pd.read_csv(labels, encoding='utf-8-sig', dtype=str)
    except Exception as e:
        df = pd.read_csv(labels, sep=';', encoding='utf-8-sig', dtype=str)

    # Number of unique identifiers
    n_unique_identifiers = df['Identifier'].nunique()
    logging.info("\n \n \n #################################\n\n")
    logging.info(f"Number of unique identifiers: {n_unique_identifiers}")

    # Get all antibiotic names (columns with and without 'CMI_')
    antibiotic_names = []
    for col in df.columns:
        if col.startswith('CMI_'):
            ab_name = col[4:]
            if ab_name in df.columns:
                antibiotic_names.append(ab_name)

    # Prepare stats
    stats = {
        'n_unique_identifiers': n_unique_identifiers,
        'antibiotics': {}
    }

    ab_stats_list = []
    for ab in antibiotic_names:
        cmi_col = 'CMI_' + ab
        ab_col = ab

        # Rows with info (not null/empty) in either CMI or ab_col
        rows_with_info = df[[cmi_col, ab_col, 'Microorganism', 'Identifier']].dropna(how='all', subset=[cmi_col, ab_col])
        n_rows_with_info = len(rows_with_info)

        # Count rows with only CMI, only ab_col, and both
        only_cmi = rows_with_info[cmi_col].notna() & rows_with_info[ab_col].isna()
        only_ab = rows_with_info[ab_col].notna() & rows_with_info[cmi_col].isna()
        both = rows_with_info[cmi_col].notna() & rows_with_info[ab_col].notna()
        n_only_cmi = int(only_cmi.sum())
        n_only_ab = int(only_ab.sum())
        n_both = int(both.sum())

        # Value counts for each column separately
        cmi_value_counts = rows_with_info.loc[rows_with_info[cmi_col].notna(), cmi_col].value_counts().to_dict()
        ab_value_counts = rows_with_info.loc[rows_with_info[ab_col].notna(), ab_col].value_counts().to_dict()

        # Species extraction
        rows_with_info = rows_with_info.copy()
        rows_with_info['species'] = rows_with_info['Microorganism'].str.split().str[:2].str.join(' ')
        species_with_info = rows_with_info['species'].unique()
        n_species_with_info = len(species_with_info)

        # Per-species stats (collect as list for sorting)
        species_stats_list = []
        for sp in species_with_info:
            sp_rows = rows_with_info[rows_with_info['species'] == sp]
            sp_only_cmi = sp_rows[cmi_col].notna() & sp_rows[ab_col].isna()
            sp_only_ab = sp_rows[ab_col].notna() & sp_rows[cmi_col].isna()
            sp_both = sp_rows[cmi_col].notna() & sp_rows[ab_col].notna()
            sp_cmi_value_counts = sp_rows.loc[sp_rows[cmi_col].notna(), cmi_col].value_counts().to_dict()
            sp_ab_value_counts = sp_rows.loc[sp_rows[ab_col].notna(), ab_col].value_counts().to_dict()
            species_stats_list.append((sp, {
                'total': int(len(sp_rows)),
                'only_cmi': int(sp_only_cmi.sum()),
                'only_ab': int(sp_only_ab.sum()),
                'both': int(sp_both.sum()),
                'cmi_value_counts': {k: int(v) for k, v in sp_cmi_value_counts.items()},
                'ab_value_counts': {k: int(v) for k, v in sp_ab_value_counts.items()}
            }))
        # Sort species_stats by total descending
        species_stats_sorted = dict(sorted(species_stats_list, key=lambda x: x[1]['total'], reverse=True))

        ab_stats_list.append((ab, {
            'total': n_rows_with_info,
            'only_cmi': n_only_cmi,
            'only_ab': n_only_ab,
            'both': n_both,
            'cmi_value_counts': {k: int(v) for k, v in cmi_value_counts.items()},
            'ab_value_counts': {k: int(v) for k, v in ab_value_counts.items()},
            'species_with_info': int(n_species_with_info),
            'species_stats': species_stats_sorted
        }))

    # Sort antibiotics by total descending
    stats['antibiotics'] = dict(sorted(ab_stats_list, key=lambda x: x[1]['total'], reverse=True))

    # Write stats to JSON file
    stats_path = os.path.join(os.path.dirname(labels), 'stats', 'amr_stats.json')
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"AMR statistics saved to {stats_path}")

def main():

    # Define base directory and file paths
    base_dir = '/MARISMa'
    raw_labels = f'{os.path.dirname(base_dir)}/RAW_MaldiMaranon/raw_AMR_mod.csv'
    labels = f'{base_dir}/AMR_labels.csv'

    # Setup logging
    log_dir = os.path.join(base_dir, 'logs')
    setup_logging(log_dir, 'amr_logger.log')

    # Match and label AMR
    # match_and_label_amr(raw_labels, labels, base_dir)

    # Get AMR statistics
    labels_cleaned = f'{base_dir}/AMR_labels_cleaned.csv'
    get_amr_statistics(labels_cleaned)


if __name__ == "__main__":
    main()
