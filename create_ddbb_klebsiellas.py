import os
import shutil
import csv

def copy_klebsiella_folders(source_base, dest_base, csv_output):
    """
    Copies the Klebsiella species folders into a database structure and generates a CSV file with metadata.

    :param source_base: The base directory containing the year folders.
    :param dest_base: The destination directory for the database.
    :param csv_output: Path for the output CSV file.
    """
    # Specify valid year folders
    valid_years = {"2018", "2019", "2020", "2021", "2022", "2023"}

    # Prepare CSV file
    with open(csv_output, mode="w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Extern_id", "Target_position", "Replicate", "Genus", "Species", "Fid_path"])

        if not os.path.exists(dest_base):
            os.makedirs(dest_base)

        # Iterate through the valid year folders
        for year in valid_years:
            year_path = os.path.join(source_base, year, "matched_bacteria")
            if os.path.isdir(year_path):  # Ensure matched_bacteria directory exists
                klebsiella_path = os.path.join(year_path, "Klebsiella")
                if os.path.exists(klebsiella_path):  # Check if Klebsiella folder exists
                    for species_folder in os.listdir(klebsiella_path):
                        species_src = os.path.join(klebsiella_path, species_folder)
                        species_dest = os.path.join(dest_base, species_folder)

                        if not os.path.exists(species_dest):
                            shutil.copytree(species_src, species_dest)
                            print(f"Copied {species_src} to {species_dest}")
                        
                        # Iterate over Extern_id folders
                        for extern_id in os.listdir(species_src):
                            extern_id_src = os.path.join(species_src, extern_id)
                            extern_id_dest = os.path.join(species_dest, extern_id)

                            if os.path.isdir(extern_id_src):
                                if not os.path.exists(extern_id_dest):
                                    shutil.copytree(extern_id_src, extern_id_dest)
                                    print(f"Copied {extern_id_src} to {extern_id_dest}")

                                # Iterate over Target Position folders
                                for position in os.listdir(extern_id_src):
                                    position_src = os.path.join(extern_id_src, position)
                                    position_dest = os.path.join(extern_id_dest, position)

                                    if os.path.isdir(position_src):
                                        if not os.path.exists(position_dest):
                                            shutil.copytree(position_src, position_dest)
                                            print(f"Copied {position_src} to {position_dest}")

                                        # Iterate over Replicate folders
                                        for replicate in os.listdir(position_src):
                                            replicate_src = os.path.join(position_src, replicate)
                                            replicate_dest = os.path.join(position_dest, replicate)

                                            if os.path.isdir(replicate_src):
                                                if not os.path.exists(replicate_dest):
                                                    shutil.copytree(replicate_src, replicate_dest)
                                                    print(f"Copied {replicate_src} to {replicate_dest}")

                                                # Construct the final fid file path
                                                fid_path = os.path.join(replicate_dest, "1SLin", "fid")

                                                # Save the row in CSV
                                                csv_writer.writerow([
                                                    extern_id,  # Extern_id
                                                    position,  # Position
                                                    replicate,  # Replicate
                                                    "Klebsiella",  # Genus
                                                    species_folder,  # Species
                                                    fid_path  # Path
                                                ])


# Define source and destination paths
source_base = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2"  # Replace with the path to your source directory
dest_base = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/DDBB_KLEBSIELLAS"
csv_output = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/DDBB_KLEBSIELLAS/klebsiella_database.csv"

# source_base = r"X:\bacteria_id\RAW_MaldiMaranon\data_cleaner_results_v2"  # Replace with the path to your source directory
# dest_base = r"X:\bacteria_id\RAW_MaldiMaranon\data_cleaner_results_v2\DDBB_KLEBSIELLAS"
# csv_output = r"X:\bacteria_id\RAW_MaldiMaranon\data_cleaner_results_v2\DDBB_KLEBSIELLAS\klebsiella_database.csv"


# Run the function
copy_klebsiella_folders(source_base, dest_base, csv_output) 