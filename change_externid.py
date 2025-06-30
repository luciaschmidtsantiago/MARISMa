import os
import re
import json
from collections import defaultdict

# CONFIGURATION
ROOT_DIR = "/export/data_ml4ds/bacteria_id/MaldiMaranonDB"
MAPPING_FILE = os.path.join(ROOT_DIR, "study_renaming_log.json")
STATS_FILE = os.path.join(ROOT_DIR, "detailed_dataset_statistics.json")
DRY_RUN = False  # Set to True to preview changes without renaming

# Match 8-digit folder names
EIGHT_DIGIT_PATTERN = re.compile(r"^\d{8}$")

# Store mapping: {old_name: new_name}
renaming_log = {}

# Track number of studies per (year, genus, species)
study_counts = defaultdict(int)

def sanitize(name):
    return name.strip().replace(" ", "_")

# ---------------------- RENAMING SECTION ----------------------
for year in os.listdir(ROOT_DIR):
    year_path = os.path.join(ROOT_DIR, year)
    if not os.path.isdir(year_path) or not year.isdigit():
        continue

    for genus in os.listdir(year_path):
        genus_path = os.path.join(year_path, genus)
        if not os.path.isdir(genus_path):
            continue

        for species in os.listdir(genus_path):
            species_path = os.path.join(genus_path, species)
            if not os.path.isdir(species_path):
                continue

            for study_folder in os.listdir(species_path):
                study_path = os.path.join(species_path, study_folder)

                if os.path.isdir(study_path) and EIGHT_DIGIT_PATTERN.match(study_folder):
                    key = (year, genus, species)
                    study_counts[key] += 1
                    index = study_counts[key]
                    new_name = f"{year}_{sanitize(genus)}_{sanitize(species)}_{index}"
                    new_path = os.path.join(species_path, new_name)

                    if DRY_RUN:
                        print(f"[DRY RUN] Would rename: {study_path} -> {new_path}")
                    else:
                        try:
                            os.rename(study_path, new_path)
                            renaming_log[study_folder] = new_name
                            print(f"Renamed: {study_path} -> {new_path}")
                        except Exception as e:
                            print(f"Failed to rename {study_path}: {e}")

# Save renaming log
with open(MAPPING_FILE, "w") as f:
    json.dump(renaming_log, f, indent=4)
print(f"\nRenaming completed. Log saved to {MAPPING_FILE}")

# ---------------------- STATISTICS SECTION ----------------------

print("\nGenerating dataset statistics...")

# Stats structure: nested dictionaries
detailed_stats = {}

for year in os.listdir(ROOT_DIR):
    year_path = os.path.join(ROOT_DIR, year)
    if not os.path.isdir(year_path) or not year.isdigit():
        continue

    year_stats = {
        "total_spectra": 0,
        "num_genera": 0,
        "num_species": 0,
        "num_studies": 0,
        "genera": {}
    }

    for genus in os.listdir(year_path):
        genus_path = os.path.join(year_path, genus)
        if not os.path.isdir(genus_path):
            continue

        genus_stats = {
            "total_spectra": 0,
            "num_species": 0,
            "num_studies": 0,
            "species": {}
        }

        for species in os.listdir(genus_path):
            species_path = os.path.join(genus_path, species)
            if not os.path.isdir(species_path):
                continue

            num_studies = 0
            num_spectra = 0

            for study_folder in os.listdir(species_path):
                study_path = os.path.join(species_path, study_folder)
                if not os.path.isdir(study_path):
                    continue

                num_studies += 1
                # Count biological replicates (target positions)
                for target_pos in os.listdir(study_path):
                    target_path = os.path.join(study_path, target_pos)
                    if not os.path.isdir(target_path):
                        continue

                    # Count technical replicates
                    for tech_rep in os.listdir(target_path):
                        tech_path = os.path.join(target_path, tech_rep)
                        if os.path.isdir(tech_path):
                            num_spectra += 1

            # Update species level
            genus_stats["species"][species] = {
                "num_studies": num_studies,
                "spectra": num_spectra
            }

            # Update genus/year level
            genus_stats["num_species"] += 1
            genus_stats["num_studies"] += num_studies
            genus_stats["total_spectra"] += num_spectra

            year_stats["num_species"] += 1
            year_stats["num_studies"] += num_studies
            year_stats["total_spectra"] += num_spectra

        year_stats["num_genera"] += 1
        year_stats["genera"][genus] = genus_stats

    detailed_stats[year] = year_stats

# Save statistics file
with open(STATS_FILE, "w") as f:
    json.dump(detailed_stats, f, indent=4)

print(f"Dataset statistics saved to {STATS_FILE}")