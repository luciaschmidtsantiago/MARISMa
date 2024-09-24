import os
import shutil
import pandas as pd
import aspose.zip as az


def remove_empty_files_from_zip(zip_file_path):
    # Load the ZIP archive
    archive = az.Archive(zip_file_path)
    entries_to_delete = []
    fid_count = 0

    # Loop through ZIP entries
    for entry in archive.entries:
        if entry.uncompressed_size == 0:
            entries_to_delete.append(entry)
        elif os.path.basename(entry.name) == "fid":
            fid_count += 1

    # Delete all listed entries
    for entry in entries_to_delete:
        archive.delete_entry(entry)

    # Save updated ZIP archive
    archive.save(zip_file_path)

    return fid_count

def postprocess(folder_path):
    data = []
    total_fid_count = 0

    # Iterate through all directories and files in the base directory
    for root, dir, files in os.walk(folder_path):

        species_fid_count = 0
        for folder, _, files in os.walk(root):
            for file in files:
                if file.startswith("fid"):
                    fid_path = os.path.join(folder, file)
                    print(f"Found 'fid' file: {fid_path}")
                    species_fid_count += 1

                # data.append([genus, species, fid_count_species])
                # total_fid_count += fid_count_species

    return data, total_fid_count

def main():
    base_dir = "Z:/lschmidt/MaldiMaranonDB"
    # base_dir = "/export/usuarios_ml4ds/lschmidt/MaldiMaranonDB"

    data, total_fid_count = postprocess(base_dir)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["Genus", "Species", "fid_count"])
    genus_sum = df.groupby("Genus")["fid_count"].sum().reset_index()
    genus_sum.columns = ["Genus", "total_fid_count"]

    # Print the DataFrames
    print(df)
    print(genus_sum)
    print(f"Total 'fid' files in the dataset: {total_fid_count}")

    # Save the DataFrame to a CSV file
    df.to_csv("fid_counts.csv", index=False)
    genus_sum.to_csv("genus_fid_counts.csv", index=False)

    # Save the DataFrame to an Excel file
    with pd.ExcelWriter("fid_counts.xlsx") as writer:
        df.to_excel(writer, sheet_name="Species", index=False)
        genus_sum.to_excel(writer, sheet_name="Genus", index=False)

if __name__ == "__main__":
    main()
