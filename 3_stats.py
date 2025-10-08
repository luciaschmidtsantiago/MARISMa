## Code by Inés López-Mareca and Lucía Schmidt-Santiago. Copyrght.

import os
import json
from tqdm import tqdm
import pandas as pd
import argparse

def collect_stats(base_dir, years):
    """Collects detailed statistics from the dataset and saves a CSV of identifiers.
     Args:
    base_dir (str): Base directory containing year subdirectories.
    years (list): List of years to process.
    Returns:
    dict: Detailed statistics per year."""

    detailed_stats = {}
    rows = []

    total_genera_set = set()
    total_species_set = set()

    for year in years:
        print(f"Processing year: {year}")

        year_stats = {
            'total_spectra': 0,
            'num_genera': 0,
            'num_species': 0,
            'num_studies': 0,
            'genera': {}
        }
        genus_species_set = set()
        year_studies_set = set()
        year_genera_set = set()
        year_dir = os.path.join(base_dir, str(year))

        for root, dirs, files in tqdm(os.walk(year_dir), desc=f"Year {year}"):
            if 'fid' in files:
                parts = root.split(os.sep)
                genus = parts[-6]
                species = parts[-5]
                study = parts[-4] #identifier
                target_position = parts[-3]
                path = os.path.join(root.split(target_position)[0], target_position)

                # Collect row for CSV
                rows.append({
                    'identifier': study,
                    'target_position': target_position,
                    'year': year,
                    'genus': genus,
                    'species': species,
                    'path': path
                })

                # Update stats
                year_stats['total_spectra'] += 1
                genus_dict = year_stats['genera'].setdefault(genus, {
                    'total_spectra': 0,
                    'num_species': 0,
                    'num_studies': 0,
                    'species': {}
                })
                genus_dict['total_spectra'] += 1
                if species not in genus_dict['species']:
                    genus_dict['num_species'] += 1
                    genus_dict['species'][species] = {'spectra': 0, 'study_set': set()}
                genus_dict['species'][species]['spectra'] += 1
                genus_dict['species'][species]['study_set'].add(study)

                genus_species_set.add((genus, species))
                year_studies_set.add(study)
                year_genera_set.add(genus)

        # Finalize study counts and remove study_set
        # Sort species alphabetically
        for genus in sorted(year_stats['genera']):
            genus_dict = year_stats['genera'][genus]
            genus_dict['num_studies'] = len(set().union(
                *(s['study_set'] for s in genus_dict['species'].values())
            ))
            sorted_species = {}
            for species in sorted(genus_dict['species']):
                sp = genus_dict['species'][species]
                sorted_species[species] = {
                    'num_studies': len(sp['study_set']),
                    'spectra': sp['spectra']
                }
            genus_dict['species'] = sorted_species

        year_stats['genera'] = {g: year_stats['genera'][g] for g in sorted(year_stats['genera'])}
        year_stats['num_genera'] = len(year_genera_set)
        year_stats['num_species'] = len(genus_species_set)
        year_stats['num_studies'] = len(year_studies_set)
        detailed_stats[str(year)] = year_stats

        total_genera_set.update(year_genera_set)
        total_species_set.update(genus_species_set)

    # Totales generales reales (únicos)
    total_spectra = sum(y['total_spectra'] for y in detailed_stats.values())
    total_studies = sum(y['num_studies'] for y in detailed_stats.values())
    total_genera = len(total_genera_set)
    total_species = len(total_species_set)

    detailed_stats['total'] = {
        'total_spectra': total_spectra,
        'num_studies': total_studies,
        'num_genera': total_genera,
        'num_species': total_species
    }

    return detailed_stats, rows, total_species_set

def save_genus_stats(stats, save_path):
    """Aggregate genus-level statistics and save to JSON."""
    genus_stats = {}
    for year, year_data in stats.items():
        if year == 'total':
            continue
        for genus, genus_data in year_data['genera'].items():
            gstat = genus_stats.setdefault(genus, {
                'total_spectra': 0,
                'total_species': set(),
                'total_studies': set(),
                'years': {},
                'species': {}
            })
            gstat['total_spectra'] += genus_data['total_spectra']
            gstat['total_species'].update(genus_data['species'].keys())
            # Collect all study identifiers for each species
            for species, sp_data in genus_data['species'].items():
                sstat = gstat['species'].setdefault(species, {
                    'total_spectra': 0,
                    'total_studies': 0,
                    'study_set': set()
                })
                sstat['total_spectra'] += sp_data['spectra']
                sstat['total_studies'] += sp_data['num_studies']
                # Add study identifiers if available
                if 'study_set' in sp_data:
                    sstat['study_set'].update(sp_data['study_set'])
            # Per year
            gstat['years'][year] = {
                'total_spectra': genus_data['total_spectra'],
                'total_species': genus_data['num_species'],
                'total_studies': genus_data['num_studies']
            }
    # Convert sets to counts for JSON
    genus_stats_json = {}
    for genus, gstat in genus_stats.items():
        # total_studies: sum of all unique studies in all species for this genus
        all_studies = set()
        for sstat in gstat['species'].values():
            all_studies.update(sstat.get('study_set', set()))
        # Remove study_set from species for JSON output
        species_json = {}
        for species, sstat in gstat['species'].items():
            species_json[species] = {
                'total_spectra': sstat['total_spectra'],
                'total_studies': sstat['total_studies']
            }
        genus_stats_json[genus] = {
            'total_spectra': gstat['total_spectra'],
            'total_species': len(gstat['total_species']),
            'total_studies': len(all_studies),
            'years': gstat['years'],
            'species': species_json
        }
    genus_json_path = os.path.join(save_path, 'genus_stats.json')
    with open(genus_json_path, 'w') as f:
        json.dump(genus_stats_json, f, indent=4)
    print(f"Genus-level statistics saved to {genus_json_path}")

def write_species_spectra_ranking(stats, save_path):
    """
    Write ranking of species by spectra count to a TXT file.
    Args:
        stats (dict): Detailed statistics as returned by collect_stats.
        save_path (str): Directory to save the ranking file.
    """
    species_spectra = {}
    for year, year_data in stats.items():
        if year == 'total':
            continue
        for genus, genus_data in year_data['genera'].items():
            for species, sp_data in genus_data['species'].items():
                key = f"{genus} {species}"
                species_spectra[key] = species_spectra.get(key, 0) + sp_data['spectra']
    ranked = sorted(species_spectra.items(), key=lambda x: x[1], reverse=True)
    ranking_path = os.path.join(save_path, 'species_spectra_ranking.txt')
    with open(ranking_path, 'w') as f:
        f.write("Species ranking by spectra count:\n")
        for i, (sp, count) in enumerate(ranked, 1):
            f.write(f"{i}. {sp}: {count}\n")
    print(f"Species spectra ranking saved to {ranking_path}")

def main(base_dest_dir, years):
    
    stats, rows, total_species_set = collect_stats(base_dest_dir, years)

    save_path = os.path.join(base_dest_dir, 'stats')
    os.makedirs(save_path, exist_ok=True)

    # Save detailed stats as JSON
    stats_path = os.path.join(save_path, 'detailed_dataset_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Detailed statistics saved to {stats_path}")

    # Save identifiers to CSV
    df = pd.DataFrame(rows)
    df.drop_duplicates(inplace=True)
    df.sort_values(by='identifier', inplace=True)
    csv_path = os.path.join(save_path,'valid_spectra.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # Save unique genera and species names to TXT (flat lists)
    unique_genera = sorted({genus for genus, _ in total_species_set})
    unique_species = sorted({species for _, species in total_species_set})

    unique_taxa_path = os.path.join(save_path, 'unique_taxonomy.txt')
    with open(unique_taxa_path, 'w') as f:
        f.write("Genera: " + ", ".join(unique_genera) + "\n")
        f.write("Species: " + ", ".join(unique_species) + "\n")
    print(f"Unique genera and species saved to {unique_taxa_path}")

    # Aggregate genus-level stats and save to CSV
    save_genus_stats(stats, save_path)

    # Rank species by spectra count and save to TXT
    write_species_spectra_ranking(stats, save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MARISMa checker script.")
    parser.add_argument("--base_dest_dir", type=str, default='/MARISMa', help="Base destination directory for output")
    parser.add_argument("years", nargs='+', type=int, help="Year(s) to process (e.g. 2019 2020)")
    args = parser.parse_args()

    print(f"Base destination directory: {args.base_dest_dir}")
    print(f"Years to process: {args.years}")

    main(args.base_dest_dir, args.years)