import os

# Root of your renamed dataset (e.g., bacteria_id)
ROOT_DIR = "/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

matches = []

for year in os.listdir(ROOT_DIR):
    year_path = os.path.join(ROOT_DIR, year)
    if not os.path.isdir(year_path):
        continue

    for genus in os.listdir(year_path):
        genus_path = os.path.join(year_path, genus)
        if not os.path.isdir(genus_path):
            continue

        for species in os.listdir(genus_path):
            species_path = os.path.join(genus_path, species)
            if not os.path.isdir(species_path):
                continue

            for study in os.listdir(species_path):
                study_path = os.path.join(species_path, study)
                if not os.path.isdir(study_path):
                    continue

                for target_pos in os.listdir(study_path):
                    target_path = os.path.join(study_path, target_pos)
                    if not os.path.isdir(target_path):
                        continue

                    # Look for "2" inside the target position folder
                    if "2" in os.listdir(target_path):
                        full_path = os.path.join(target_path, "2")
                        matches.append(full_path)
                        print(f"Found technical replicate '2': {full_path}")

print(f"\nFound {len(matches)} samples with a technical replicate '2'.")