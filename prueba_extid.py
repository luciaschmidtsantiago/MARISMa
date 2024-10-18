import os
import zipfile
import pandas as pd
import json
import xml.etree.ElementTree as ET
import logging
import re

# Initialize a dictionary to store genera, species, and their respective counts
data_dict = {}

def setup_logging(year):
    """
    Sets up logging with file names including the specified year.

    :param year: The year to include in the log file names.
    """
    # Logger for found FID and ACQUS files
    fid_acqus_log = logging.getLogger('fid_acqus')
    fid_acqus_log_handler = logging.FileHandler(f'fid_acqus_log_{year}.txt')
    fid_acqus_log.addHandler(fid_acqus_log_handler)

    # Logger for recording matches in XML files
    xml_match_log = logging.getLogger('xml_match')
    xml_match_log_handler = logging.FileHandler(f'xml_match_log_{year}.txt')
    xml_match_log.addHandler(xml_match_log_handler)

    # Logger to describe the matching process
    match_process_log = logging.getLogger('match_process')
    match_process_log_handler = logging.FileHandler(f'match_process_log_{year}.txt')
    match_process_log.addHandler(match_process_log_handler)

    # Logger for recording errors that occur during processing
    error_log = logging.getLogger('error_log')
    error_log_handler = logging.FileHandler(f'error_log_{year}.txt')
    error_log.addHandler(error_log_handler)

    # Set log level and format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Return the loggers for use in the main code
    return fid_acqus_log, xml_match_log, match_process_log, error_log



def explore_directories(zip_path, year, fid_acqus_log, xml_match_log, match_process_log, error_log, processed_dirs=set()):
    """
    Explores the directories in the provided ZIP file to find FID and ACQUS files.

    :param zip_path: The path to the ZIP file.
    :param processed_dirs: Set of directories already processed to avoid repetition.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Iterate over the list of files in the ZIP
            for file in zip_ref.namelist():
                # Check if the file is a directory
                if file.endswith('/'):
                    continue  # Skip directories

                # Get the directory path (e.g., the folder containing FID and ACQUS)
                dir_path = os.path.dirname(file)

                # Check if the directory has already been processed
                if dir_path in processed_dirs:
                    continue  # Skip this directory if it has been processed

                # Check if the file is a FID and if the corresponding ACQUS exists
                if file.lower().endswith('fid') and f"{dir_path}/ACQUS".lower() in map(str.lower, zip_ref.namelist()):
                    fid_acqus_log.info(f"FID and ACQUS files found in: {dir_path}")
                    print(f"FID and ACQUS files found in: {dir_path}")

                    # Extract extern_id and target_position from the directory structure
                    path_parts = dir_path.split('/')
                    if len(path_parts) >= 3:
                        global extern_id, target_position
                        extern_id = path_parts[-4]  # Third level for externId

                        # Check if extern_id has at least 8 characters and a digit
                        if len(extern_id) >= 8 and extern_id[:8].isdigit():
                            extern_id = extern_id[:8]  # Take the first 8 characters
                        else:
                            continue  # Skip if extern_id has fewer than 8 characters
                        
                        

                        target_position = path_parts[-3].split('_')[1]  # First part of the name for targetPosition

                        # Mark the directory as processed
                        processed_dirs.add(dir_path)

                        # Search for matches in the XML files
                        search_in_xml(extern_id, target_position, year, fid_acqus_log, xml_match_log, match_process_log, error_log)

    except Exception as e:
        error_log.error(f"Error exploring directories: {str(e)}")


def search_in_xml(extern_id, target_position, year, fid_acqus_log, xml_match_log, match_process_log, error_log):
    """
    Searches for matches in the XML files for the given extern_id and target_position.

    :param extern_id: The externId to search for.
    :param target_position: The targetPosition to search for.
    :param year: The year of the files being processed.
    """
    try:
        # Define the path for the XML ZIP file
        xml_zip_path = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/XML_2018a2022.zip'
        year_folder = str(year)  # Convert the year to string for folder name
        year_folder_path = f"{year_folder}/"  # e.g., "2018/"

        with zipfile.ZipFile(xml_zip_path, 'r') as zip_ref:
            # Check if the specified year folder exists in the ZIP
            all_files = zip_ref.namelist()
            
            files_in_year_folder = [file for file in all_files if file.startswith(year_folder_path)]

            if not files_in_year_folder:
                error_log.error(f"Year folder '{year_folder_path}' does not exist in the ZIP or is empty.")
                return

            # Iterate through the files in the ZIP, filtering by the specific year folder
            for file in files_in_year_folder:
                # Only process XML files within the specified year folder
                if file.startswith(year_folder_path) and file.endswith('.xml'):
                    with zip_ref.open(file) as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()

                        for analyte in root.findall(".//Analyte"):
                            extern_id_xml = analyte.attrib.get('externId', '')[:8]

                            # Check if extern_id_xml is at least 8 characters and is numeric
                            if len(extern_id_xml) < 8 or not extern_id_xml.isdigit():
                                continue

                            target_position_xml = analyte.attrib.get('targetPosition')

                            # If externId and targetPosition match, process the match
                            if extern_id_xml == extern_id and target_position_xml == target_position:
                                xml_match_log.info(f"✔️ Match found in XML file: {file} for externId={extern_id} and targetPosition={target_position}")
                                print(f"✔️ Match found in XML file: {file} for externId={extern_id} and targetPosition={target_position}")

                                # Call the function to process matches
                                process_msp_matches(analyte, file, fid_acqus_log, xml_match_log, match_process_log, error_log)

                                # Once a match is found, no need to process further files
                                return

            # If no match was found after iterating through all files, log the result
            xml_match_log.info(f"❌ No match found for externId={extern_id} and targetPosition={target_position} in year {year}")
            print(f"❌ No match found for externId={extern_id} and targetPosition={target_position} in year {year}")

    except Exception as e:
        error_log.error(f"Error searching for matches in XML: {str(e)}")


# Counter to track how many results have been processed
results_counter = 0

def process_msp_matches(analyte, file, fid_acqus_log, xml_match_log, match_process_log, error_log):
    global results_counter  # To keep track of how many results have been processed
    """
    Processes matches found in the MSP matches within the Analyte.

    :param analyte: The Analyte element containing match information.
    :param file: The XML file where the Analyte was found.
    """
    try:
        if results_counter >= 10:
            return  # Stop processing once 10 results have been processed

        THRESHOLD = 1.7  # Threshold for global match value
        msp_matches = analyte.findall(".//MspMatch")[:3]  # Get the top 3 matches

        extern_id = analyte.attrib.get('externId').split('-')[0]
        target_position = analyte.attrib.get('targetPosition')

        # Validate extern_id and target_position
        if not extern_id or not target_position or not msp_matches:
            match_process_log.info(f"❌ extern_id = {extern_id} in {target_position} denied in {file}: Missing attributes")
            return

        if len(extern_id) < 8 or not extern_id[:8].isdigit():
            match_process_log.info(f"❌ extern_id = {extern_id} in {target_position} denied in {file}: Invalid extern_id")
            return

        # Process the first MSP match
        ref_pattern_1 = msp_matches[0].attrib.get("referencePatternName")
        global_match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))

        if global_match_value_1 < THRESHOLD:
            match_process_log.info(f"❌ extern_id = {extern_id} in {target_position} denied in {file}: Low globalMatchValue ({global_match_value_1})")
            return

        # Extract genus and species from the reference pattern
        genus_1, species_1 = ref_pattern_1.split()[:2]
        species_1 = re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_1).split(" ")[0]  # Clean the species name

        # Continue processing the second and third matches similarly...
        
        # Update the data_dict with the genus and species counts
        if genus_1 not in data_dict:
            data_dict[genus_1] = {}
        if species_1 not in data_dict[genus_1]:
            data_dict[genus_1][species_1] = 0

        # Increment the count for the identified species
        data_dict[genus_1][species_1] += 1

        # Log successful processing of the matches
        match_process_log.info(f"✔️ Successfully processed extern_id = {extern_id} in {target_position} from {file}: {genus_1} {species_1} found. Count: {data_dict[genus_1][species_1]}")

        # Increment the results counter
        results_counter += 1

        # If 10 results have been processed, save the results immediately
        if results_counter == 10:
            save_results(data_dict, year)

    except Exception as e:
        error_log.error(f"Error processing MSP matches: {str(e)}")

# Main function to run the script
def main(zip_path, year):
    fid_acqus_log, xml_match_log, match_process_log, error_log = setup_logging(year)

    explore_directories(zip_path, year, fid_acqus_log, xml_match_log, match_process_log, error_log)

    save_results(data_dict, year)  # Save the results obtained in CSV, JSON, and XML

# Function to save results in different formats
def save_results(data, year):
    """
    Saves the results in CSV, JSON, and XML.

    :param data: The dictionary containing the results.
    :param year: The year entered by the user.
    """
    # Save as CSV
    df = pd.DataFrame.from_dict({(i,j): data[i][j] 
                                 for i in data.keys() 
                                 for j in data[i].keys()},
                                 orient='index')
    df.to_csv(f'results_{year}.csv', header=['Count'], index_label=['Genus', 'Species'])

    # Save as JSON
    with open(f'results_{year}.json', 'w') as json_file:
        json.dump(data, json_file)

    # Save as XML
    root = ET.Element("Results")
    for genus, species_dict in data.items():
        genus_elem = ET.SubElement(root, "Genus", name=genus)
        for species, count in species_dict.items():
            ET.SubElement(genus_elem, "Species", name=species).text = str(count)

    tree = ET.ElementTree(root)
    tree.write(f'results_{year}.xml')


if __name__ == "__main__":
    year = input("Please enter the year you want to process (e.g., 2020): ")

    # Generate the ZIP file path dynamically with the entered year
    zip_path = f"/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/{year}.zip"
    

    # Execute the processing
    main(zip_path, year)
