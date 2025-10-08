## Code by Inés López-Mareca and Lucía Schmidt-Santiago. Copyrght.

import os
import re
import json
import logging
import zipfile
import argparse
import pandas as pd
import xml.etree.ElementTree as ET

from collections import defaultdict


THRESHOLD = 1.7

def setup_logging(log_directory, year):
    """
    Sets up logging for the script, creating a directory structure based on the current date and year.
    Creates separate log files for different types of logs.
    :param log_directory: The directory where the log files will be stored.
    :param year: The year for which the logs are being created.
    :return: A tuple of loggers for different log types.
    """

    # Setup individual loggers for different logs
    xml_found_log = logging.getLogger('xml_found_log')
    xml_found_log_handler = logging.FileHandler(os.path.join(log_directory, f'xml_found_log_{year}.log'))
    xml_found_log.addHandler(xml_found_log_handler)

    analyte_validation_log = logging.getLogger('analyte_validation_log')
    analyte_validation_log_handler = logging.FileHandler(os.path.join(log_directory, f'analyte_validation_log_{year}.log'))
    analyte_validation_log.addHandler(analyte_validation_log_handler)
    
    analyte_zip_found_log = logging.getLogger('analyte_zip_found_log')
    analyte_zip_found_log_handler = logging.FileHandler(os.path.join(log_directory, f'analyte_zip_found_log_{year}.log'))
    analyte_zip_found_log.addHandler(analyte_zip_found_log_handler)
    
    directory_move_log = logging.getLogger('directory_move_log')
    directory_move_log_handler = logging.FileHandler(os.path.join(log_directory, f'directory_move_log_{year}.log'))
    directory_move_log.addHandler(directory_move_log_handler)

    # Logger for recording errors that occur during processing
    error_log = logging.getLogger('error_log')
    error_log_handler = logging.FileHandler(os.path.join(log_directory, f'error_log_{year}.log'))
    error_log.addHandler(error_log_handler)

    # Logger for recording errors that occur during processing
    extern_id_op_log = logging.getLogger('extern_id_op_log')
    extern_id_op_log_handler = logging.FileHandler(os.path.join(log_directory, f'extern_id_op_log_{year}.log'))
    extern_id_op_log.addHandler(extern_id_op_log_handler)

    normalize_log = logging.getLogger('normalize_log')
    normalize_log_handler = logging.FileHandler(os.path.join(log_directory, f'normalized_entries_log_{year}.log'))
    normalize_log.addHandler(normalize_log_handler)

    # Set log level and format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Return the loggers for use in the main code
    return xml_found_log, analyte_validation_log, analyte_zip_found_log, directory_move_log, error_log, extern_id_op_log, normalize_log

def process_extern_id(extern_id, extern_id_op_log, analyte_validation_log, suffix=''):
    """
    Processes the extern_id to ensure it meets the expected format and length.
    :param extern_id: The extern_id to process.
    :param extern_id_op_log: Logger for extern ID operations.
    :param analyte_validation_log: Logger for analyte validation.
    :param suffix: Optional suffix to append to the extern_id.
    :return: Processed extern_id or None if validation fails.
    """

    # Get length of identifier
    length = len(extern_id)

    # Validation by length
    if length > 10:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} with more than 10 digits is not expected")
        return None

    # Validation of endig for identifiers of 9 >= length >= 10
    if length in (9, 10):
        if extern_id[-2:] not in {'01', '02', '03', '04', '05'}:
            analyte_validation_log.info(f"extern_id = {extern_id} with {length} digits ends with unexpected digits {extern_id[-2:]}")
            return None

    # Transformación del ID
    if length == 7:
        new_id = '0' + extern_id + suffix
        extern_id_op_log.info(f"⚠️ Operation: Padding with 0 in front of 7 digits --> {extern_id} -> {new_id}")
    elif length == 8:
        new_id = extern_id + suffix
        extern_id_op_log.info(f"✅ Operation: No change for 8 digits --> {extern_id} -> {new_id}")
    elif length == 9:
        new_id = '0' + extern_id[:-2] + suffix
        extern_id_op_log.info(f"⚠️ Operation:  its --> {extern_id} -> {new_id}")
    elif length == 10:
        new_id = extern_id[:-2] + suffix
        extern_id_op_log.info(f"⚠️ Operation: Removed last 2 digits from 10 digits --> {extern_id} -> {new_id}")
    else:
        extern_id_op_log.info(f"❌ Identifier of undefined length --> {extern_id}")
        return None

    return new_id

def normalize_valid_entries(valid_entries, normalize_log):
    """
    Normalizes the valid entries by grouping them based on their base extern_id and suffixes.
    Handles conflicts and unifies entries where applicable.
    :param valid_entries: List of valid entries to normalize.
    :param normalize_log: Logger for normalization operations.
    :return: List of normalized entries.
    """

    grouped = defaultdict(list)
    for genus, species, extern_id, position in valid_entries:
        match = re.match(r'^(\d{8})(.*)$', extern_id)
        if not match:
            normalize_log.info(f"⚠️ Ignored malformed extern_id: {extern_id}")
            continue
        base_id, suffix = match.groups()
        suffix = suffix if suffix else None
        grouped[base_id].append((genus, species, suffix, extern_id, position))

    normalized_entries = []
    extern_ids_to_remove = set()
    normalized_ids = set()

    for base_id, records in grouped.items():
        suffix_species = defaultdict(set)
        species_by_suffix = defaultdict(set)

        for genus, species, suffix, full_id, position in records:
            suffix_species[suffix].add(species)
            species_by_suffix[(suffix, species)].add((genus, full_id, position))

        # CASE D: no suffix, but multiple species → conflict
        if None in suffix_species and len(suffix_species[None]) > 1:
            normalize_log.info(f"❌ Conflict: base_id {base_id} has no suffix but multiple species: {suffix_species[None]}")
            for genus, species, suffix, extid, pos in records:
                if suffix is None:
                    normalize_log.info(f"  → Removed (no suffix conflict): {genus} {species} {extid} {pos}")
                    extern_ids_to_remove.add(extid)
            continue

        # CASE B: same suffix, multiple species → conflict
        for suffix, species_set in suffix_species.items():
            if len(species_set) > 1:
                normalize_log.info(f"❌ Conflict: base_id {base_id} with suffix '{suffix}' has multiple species: {species_set}")
                for species in species_set:
                    for genus, extid, pos in species_by_suffix[(suffix, species)]:
                        normalize_log.info(f"  → Removed (same suffix conflict): {genus} {species} {extid} {pos}")
                        extern_ids_to_remove.add(extid)

        # CASE C: group same-species suffixes and unify them individually
        species_to_suffixes = defaultdict(list)
        for suffix, species_set in suffix_species.items():
            if len(species_set) == 1:
                species = next(iter(species_set))
                species_to_suffixes[species].append(suffix)

        used_suffixes = set()
        for species, suffixes in species_to_suffixes.items():
            if len(suffixes) > 1:
                # Remove suffixes tha are none
                suffixes = [s for s in suffixes if s is not None]
                preferred_suffix = suffixes[0]   # Pick the first suffix as canonical
                normalize_log.info(f"✅ Normalized: base_id {base_id} for species '{species}' → unified to suffix '{preferred_suffix}'")
                for suffix in suffixes:
                    for genus, full_id, position in species_by_suffix[(suffix, species)]:
                        if full_id not in extern_ids_to_remove:
                            new_extid = base_id + preferred_suffix
                            normalized_entries.append((genus, species, new_extid, position))
                            normalized_ids.add(full_id)
                used_suffixes.update(suffixes)

        # CASE A: everything else → keep original entries unless marked for removal
        for genus, species, suffix, extid, position in records:
            if extid not in extern_ids_to_remove and extid not in normalized_ids:
                normalized_entries.append((genus, species, extid, position))
            elif extid in extern_ids_to_remove:
                normalize_log.info(f"❌ Removed: {genus} {species} {extid} at position {position} due to conflicts")

    normalize_log.info(f"\n✅ Final normalized entries count: {len(normalized_entries)}")
    return normalized_entries

def is_valid_species(name):
    """
    Validates the species name by checking if it contains only alphabetic characters and is not empty.
    :param name: The species name to validate.
    :return: The cleaned species name if valid, otherwise None."""
    name = re.sub(r'[^a-zA-Z0-9 \n\.]', ' ', name).split(" ")[0]
    return name if re.match(r'^[a-zA-Z]+$', name) else None

def parse_ref_pattern(msp_match):
    """
    Parses the reference pattern from an MspMatch element to extract genus and species.
    :param msp_match: The MspMatch XML element.
    :return: A tuple of (genus, species) if valid, otherwise None.
    """
    ref = msp_match.attrib.get("referencePatternName", "")
    parts = ref.split()
    if len(parts) < 2:
        return None
    genus, raw_species = parts[:2]
    species = is_valid_species(raw_species)
    return genus, species

def validate_extern_id(extern_id, extern_id_op_log, analyte_validation_log, target_position):
    """
    Validates the extern_id to ensure it meets the expected format and length.
    :param extern_id: The extern_id to validate.
    :param extern_id_op_log: Logger for extern ID operations.
    :param analyte_validation_log: Logger for analyte validation.
    :param target_position: The target position of the analyte.
    :return: Processed extern_id or None if validation fails.
    """
    if len(extern_id) < 7:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Too short extern_id")
        return None

    if extern_id.isdigit():
        return process_extern_id(extern_id, extern_id_op_log, analyte_validation_log)

    match = re.search(r'\D', extern_id)
    if not match:
        return None

    idx = match.start()
    num_part, suffix = extern_id[:idx], extern_id[idx:]
    if len(num_part) < 7:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Too short extern_id")
        return None

    if suffix.startswith(('-', '.')):
        return process_extern_id(num_part, extern_id_op_log, analyte_validation_log, suffix)
    return process_extern_id(num_part, extern_id_op_log, analyte_validation_log)

def validate_match(genus_1, species_1, genus_n, species_n, match_value, idx, ext_id, target_position, log):
    """
    Validates the match between the first and subsequent MspMatch elements.
    :param genus_1: Genus from the first MspMatch.
    :param species_1: Species from the first MspMatch.
    :param genus_n: Genus from the nth MspMatch.
    :param species_n: Species from the nth MspMatch.
    :param match_value: Global match value of the nth MspMatch.
    :param idx: Index of the nth MspMatch.
    :param ext_id: Extern ID of the analyte.
    :param target_position: Target position of the analyte.
    :param log: Logger for validation messages.
    :return: True if the match is valid, False otherwise."""
    if species_n is None:
        log.info(f"❌ extern_id = {ext_id} at {target_position} denied: Invalid species name ({idx}: {species_n})")
        return False

    if genus_1 != genus_n and match_value >= THRESHOLD:
        log.info(f"❌ extern_id = {ext_id} at {target_position} denied: Genus mismatch ({idx}: {genus_n}) with high global_match_value ({match_value})")
        return False

    if species_1 != species_n:
        if match_value >= THRESHOLD:
            log.info(f"❌ extern_id = {ext_id} at {target_position} denied: Species mismatch at high global_match_value ({idx}: {species_n}): {match_value}")
            return False
        else:
            log.info(f"⚠️ extern_id = {ext_id} at {target_position} accepted: Species mismatch at low global_match_value ({idx}: {species_n}): {match_value}")
    return True

def process_analyte(analyte, analyte_validation_log, extern_id_op_log):
    """
    Processes an Analyte XML element to extract and validate genus, species, extern_id, and target_position.
    :param analyte: The Analyte XML element.
    :param analyte_validation_log: Logger for analyte validation.
    :param extern_id_op_log: Logger for extern ID operations.
    :return: A tuple of (genus, species, extern_id, target_position) if the analyte is valid, otherwise None.
    """
    extern_id = analyte.attrib.get("externId")
    target_position = analyte.attrib.get("targetPosition")
    msp_matches = analyte.findall(".//MspMatch")[:3]

    if not extern_id or not target_position or len(msp_matches) < 3:
        analyte_validation_log.info(f"❌ extern_id = {extern_id} at {target_position} denied: Missing attributes")
        return None

    ext_id_processed = validate_extern_id(extern_id, extern_id_op_log, analyte_validation_log, target_position)
    if ext_id_processed is None:
        return None

    genus_1, species_1 = parse_ref_pattern(msp_matches[0])
    if species_1 is None:
        analyte_validation_log.info(f"❌ extern_id = {ext_id_processed} at {target_position} denied: Invalid species name (1: {species_1})")
        return None

    match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))
    if match_value_1 < THRESHOLD:
        analyte_validation_log.info(f"❌ extern_id = {ext_id_processed} at {target_position} denied: Low global_match_value ({match_value_1})")
        return None

    for i, msp in enumerate(msp_matches[1:], start=2):
        genus_n, species_n = parse_ref_pattern(msp)
        if species_n is None:
            analyte_validation_log.info(f"❌ extern_id = {ext_id_processed} at {target_position} denied: Invalid species name ({i}: {species_n})")
            return None
        match_value_n = float(msp.attrib.get("globalMatchValue", "0"))
        if not validate_match(genus_1, species_1, genus_n, species_n, match_value_n, i, ext_id_processed, target_position, analyte_validation_log):
            return None

    analyte_validation_log.info(f"✅ extern_id = {ext_id_processed} at {target_position} accepted: {genus_1} {species_1} with {match_value_1}")
    return genus_1, species_1, ext_id_processed, target_position

def zip_and_move_folder(zip_ref: zipfile.ZipFile, folder_name, new_dest_path, directory_move_log, error_log):
    """
    Extracts files from a zip file and moves them to a new destination path, ensuring the directory structure is maintained.
    :param zip_ref: The zip file object to extract from.
    :param folder_name: The name of the folder to extract from the zip file.
    :param new_dest_path: The destination path where the files will be moved.
    :param directory_move_log: Logger for directory move messages.
    :param error_log: Logger for error messages.
    :return: None
    """
    for src_file_path in zip_ref.namelist():
        if not src_file_path.startswith(folder_name):
            continue
        if src_file_path.endswith('/'):
            continue
        dest_file_path = os.path.join(new_dest_path, src_file_path[len(folder_name):])

        success = False
        for _ in range(3):
            try:
                # Write entry to disk
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                with open(dest_file_path, 'wb') as dest_file:
                    dest_file.write(zip_ref.read(src_file_path))
                
                # Validate destination file
                if os.path.getsize(dest_file_path) > 0:
                    success = True
                    break
                directory_move_log.info(f'❌ Failed to write {dest_file_path}')
            except Exception as e:
                error_log.info(f'❌ Error writing {dest_file_path}: {e}')
                if 'Bad CRC-32' in str(e):
                    error_log.info(f'❌ Skipping file due to CRC error: {dest_file_path}')
                    return False

        # Handle severe fails
        if not success:
            error_log.info(f'❌ Something is wrong with destination filesystem, panic: {dest_file_path}')
            error_log.info(f' → Skipping file due to repeated errors: {dest_file_path}')
            return False
        
        directory_move_log.info(f"Success: {dest_file_path}")
        
def organize_files(valid_entries, analyte_zip_found_log, directory_move_log, error_log, zip_file_path, base_dest_dir):
    """
    Organizes files based on valid entries by extracting them from a zip file and moving them to a structured directory.
    :param valid_entries: List of valid entries containing genus, species, extern_id, and target_position.
    :param analyte_zip_found_log: Logger for found analyte zip files.
    :param directory_move_log: Logger for directory move messages.
    :param error_log: Logger for error messages.
    :param zip_file_path: Path to the zip file containing the analyte data.
    :param base_dest_dir: Base destination directory where files will be organized.
    :return: DataFrame with species counts organized by genus and species.
    """
    species_count = {}
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for genus, species, extern_id, target_position in valid_entries:
            # Find the corresponding folder in the zip file
            folder_name = None
            for file in zip_ref.namelist():
                base_id = extern_id[:8]  # Use the first 8 digits of the extern_id
                if base_id in file and target_position in file:
                # if f'/{extern_id}/0_{target_position}/' in file:
                    folder_name = file
                    # if folder_name != f'{file.split(extern_id)[0]}{extern_id}/0_{target_position}/':
                    #     error_log.info(f'❌ Illegal state for {file}') # Log the error and continue
                    #     continue
                    break
            
            if not folder_name:
                analyte_zip_found_log.info(f"❌ {extern_id} at {target_position} NOT FOUND\n")
                continue
            else:
                # Cut the extern_id to the difits
                matches = [entry for entry in valid_entries if extern_id[:8] in entry[2]]
                sps = []
                # if the second position in all matches is the same, keep only the first 8 digits of the extern_id
                for match in matches:
                    sps.append(match[1])
                if len(set(sps)) == 1:
                    extern_id = extern_id[:8]
                else:
                    extern_id = extern_id.replace(".", "-")

                dest_dir = os.path.join(base_dest_dir, genus, species, extern_id, f'0_{target_position}')
                zip_and_move_folder(zip_ref, folder_name, dest_dir, directory_move_log, error_log)

                # Update species count within genus
                if genus in species_count:
                    if species in species_count[genus]:
                        species_count[genus][species] += 1
                    else:
                        species_count[genus][species] = 1
                else:
                    species_count[genus] = {species: 1}

    # Convert species count to DataFrame
    data = []
    for genus, species_dict in species_count.items():
        for species, count in species_dict.items():
            data.append({'Genus': genus, 'Species': species, 'Count': count})
    df = pd.DataFrame(data, columns=['Genus', 'Species', 'Count'])
    return df

def find_xml_files(xml_file_path, year, xml_found_log, analyte_validation_log, extern_id_op_log, error_log):
    """
    Finds and processes XML files in a zip file for a given year, extracting valid entries based on analyte data.
    :param xml_file_path: Path to the zip file containing XML files.
    :param year: Year for which the XML files are being processed.
    :param xml_found_log: Logger for found XML files.
    :param analyte_validation_log: Logger for analyte validation.
    :param extern_id_op_log: Logger for extern ID operations.
    :param error_log: Logger for error messages.
    """
    valid_entries = []
    try:
        with zipfile.ZipFile(xml_file_path, 'r') as zip_ref:
            xml_files = [file for file in zip_ref.namelist() if file.endswith('.xml')]
            
            if not xml_files:
                xml_found_log.info(f"❌ No XML file found in {xml_file_path} for year {year}\n")
                raise Exception(f'No XML file found in the zip file for year {year}')

            for file in xml_files:
                xml_found_log.info(f"✅ {file} found\n")
                with zip_ref.open(file) as f:
                    try:
                        tree = ET.parse(f)
                        root = tree.getroot()
                        for analyte in root.findall(".//Analyte"):
                            try:
                                results = process_analyte(analyte, analyte_validation_log, extern_id_op_log)
                                if results:
                                    genus, species, extern_id, target_position = results
                                    # print("Genus:", genus, "Species:", species, "Extern ID:", extern_id, "Target Position:", target_position)
                                    valid_entries.append((genus, species.capitalize(), extern_id, target_position))
                                else:
                                    # print("Invalid analyte found, skipping...")
                                    pass
                            except Exception as e:
                                error_log.info(f"❌ Error unpacking results from process_analyte: {e} at {file}\n")
                                xml_found_log.info(f"❌ Rejected due to error unpacking results from process_analyte: {e} at {file}\n")
                                continue
                    except ET.ParseError as e:
                        error_log.info(f"Error parsing XML file {file}: {e}\n")
                        xml_found_log.info(f"❌ Rejected due to error parsing XML file {file}: {e}\n")
                        continue

            if not valid_entries:
                xml_found_log.info(f"❌ No valid entries found in {xml_file_path} for year {year}\n")
                raise Exception(f'No valid entries found in the XML files for year {year}')
            
    except Exception as e:
        error_log.info(f"Error finding XML files: {e}\n")
        raise Exception(f'Error finding XML files: {e}')
    
    return valid_entries

def generate_reports(df, year, root):
    """
    Generates reports in CSV, JSON, and XML formats based on the DataFrame of species counts.
    :param df: DataFrame containing species counts organized by genus and species.
    :param year: Year for which the reports are generated.
    :param root: Root directory where the reports will be saved.
    :return: None
    """
    stats_dir = f'{root}/data_ml4ds/bacteria_id/MARISMa/stats/{year}'
    os.makedirs(stats_dir, exist_ok=True)
    
    # # Generate CSV
    # csv_path = os.path.join(stats_dir, f'report_{year}.csv')
    # df.to_csv(csv_path, index=False)
    
    # Generate JSON
    json_path = os.path.join(stats_dir, f'report_{year}.json')
    json_data = df.groupby('Genus', group_keys=False, observed=True).apply(lambda x: x.set_index('Species')['Count'].to_dict(), include_groups=False).to_dict()
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    

def main(root, base_dest_dir, years, log_dir):

    for year in years:
        print(f"Processing year: {year}")

        base_dest_dir = os.path.join(base_dest_dir, str(year))

        # Define paths
        zip_file_path = f'{root}/data_ml4ds/bacteria_id/RAW_MaldiMaranon/{year}.zip'
        xml_file_path = f'{root}/data_ml4ds/bacteria_id/RAW_MaldiMaranon/XML_{year}.zip'

        # Setup logging
        log_directory = os.path.join(log_dir, f'{year}')
        os.makedirs(log_directory, exist_ok=True)
        xml_found_log, analyte_validation_log, analyte_zip_found_log, directory_move_log, error_log, extern_id_op_log, normalize_log = setup_logging(log_directory, year)

        # Find and process XML files
        valid_entries = find_xml_files(xml_file_path, year, xml_found_log, analyte_validation_log, extern_id_op_log, error_log)
        valid_entries = normalize_valid_entries(valid_entries, normalize_log)

        valid_entries_path = os.path.join(log_directory, 'valid_entries.txt')
        os.makedirs(os.path.dirname(valid_entries_path), exist_ok=True)
        with open(valid_entries_path, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write('\t'.join(str(x) for x in entry) + '\n')
        print(f"Archivo de valid_entries guardado en: {valid_entries_path}")
        
        # Organize files based on valid entries
        df = organize_files(valid_entries, analyte_zip_found_log, directory_move_log, error_log, zip_file_path, base_dest_dir)
        
        # Generate reports
        generate_reports(df, year, root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MARISMa cleaner script.")
    parser.add_argument("root", nargs='?', type=str, default='/export', help="Set root directory: {'cpu': 'X:', 'mac': '/Volumes', 'hator': '/export'}")
    parser.add_argument("base_dest_dir", nargs='?', type=str, default='/MARISMa', help="Base destination directory for output")
    parser.add_argument("years", nargs='+', type=int, help="Year(s) to process (e.g. 2019 2020)")
    args = parser.parse_args()

    log_dir = os.path.join(args.base_dest_dir, 'logs')
    main(args.root, args.base_dest_dir, args.years, log_dir)