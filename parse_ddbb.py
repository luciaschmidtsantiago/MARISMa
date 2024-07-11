import os
import shutil
import zipfile
from tqdm import tqdm

def collect_year_xml(path, year, dest_dir):

    # Open the zip file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        
        # Get the list of files to copy
        files_to_copy = [file for file in zip_ref.namelist() if file.startswith(f"{year}/") and file.endswith(".xml")]

        # Loop through each file in the list with a progress bar
        for file in tqdm(files_to_copy, desc="Copying XML files"):
            # Read the file from the zip archive
            with zip_ref.open(file) as source:
                # Define the path where the file should be copied
                destination_path = os.path.join(dest_dir, os.path.basename(file))
                # Copy the file to the destination directory
                with open(destination_path, 'wb') as dest:
                    shutil.copyfileobj(source, dest)
    

    # Number of files copied
    files_original = len(files_to_copy)
    num_files = len([f for f in os.listdir(dest_dir) if f.endswith(".xml")])
    print(f"Number of files copied in {dest_dir}: {num_files}")
    assert files_original == num_files, f"Number of files copied ({num_files}) is different from the original ({files_original})"

    return num_files

# Define the paths
years = ["2018", "2019", "2020", "2021", "2022"]
for year in years: 
    zip_file_path = r"C:\Users\schmi\Downloads\Historico identificaciones en XML.zip"
    dest_dir = r"Z:\lschmidt\GITHUB\MaldiMaranon\xml_to_parse_15798"
    dest_dir = dest_dir + f"\{year}"

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    collect_year_xml(zip_file_path, year, dest_dir)
