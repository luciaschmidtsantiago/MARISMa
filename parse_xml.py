import os
import zipfile
import xml.etree.ElementTree as ET
import re

THRESHOLD = 1.7

def process_analyte(analyte, log):
    extern_id = analyte.attrib.get("externId")
    target_position = analyte.attrib.get("targetPosition")
    msp_matches = analyte.findall(".//MspMatch") [:3]

    if not extern_id or not target_position or not msp_matches:
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Missing attributes\n")
        return None, None, None, None
    
    if len(extern_id) < 8:
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid extern_id\n")
        return None, None, None, None

    if not extern_id[:8].isdigit():
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid extern_id\n")
        return None, None, None, None

    # Extract the referencePatternName and globalMatchValue from the first MspMatch
    ref_pattern_1 = msp_matches[0].attrib.get("referencePatternName")
    global_match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))

    if global_match_value_1 < THRESHOLD:
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Low global_match_value ({global_match_value_1})\n")
        return None, None, None, None

    if global_match_value_1 >= THRESHOLD:
        genus_1, species_1 = ref_pattern_1.split()[:2]               
        species_1 = re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_1).split(" ")[0]
        if not re.match(r'^[a-zA-Z]+$', species_1):
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (1: {species_1})\n")
            return None, None, None, None
        
        # Check the second match
        ref_pattern_2 = msp_matches[1].attrib.get("referencePatternName")
        genus_2, species_2 = ref_pattern_2.split()[:2]
        species_2 =  re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_2).split(" ")[0]
        if not re.match(r'^[a-zA-Z]+$', species_2):
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (2: {species_2})\n")
            return None, None, None, None
        global_match_value_2 = float(msp_matches[1].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_2 and global_match_value_2 >= THRESHOLD:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 2: {genus_2}) with high global_match_value ({global_match_value_2})\n")
            return None, None, None, None
        
        if species_1 != species_2 and global_match_value_2 >= THRESHOLD:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}\n")
            return None, None, None, None
        
        if species_1 != species_2 and global_match_value_2 < THRESHOLD:
            log.write(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}\n")
        
        # Check the third match
        ref_pattern_3 = msp_matches[2].attrib.get("referencePatternName")
        genus_3, species_3 = ref_pattern_3.split()[:2]
        species_3 =  re.sub('[^a-zA-Z0-9 \n\.]', ' ', species_3).split(" ")[0]
        if not re.match(r'^[a-zA-Z]+$', species_3):
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Invalid species name (3: {species_3})\n")
            return None, None, None, None
        global_match_value_3 = float(msp_matches[2].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_3 and global_match_value_3 >= THRESHOLD:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 3: {genus_3} with high global_match_value ({global_match_value_2})\n")
            return None, None, None, None

        if species_1 != species_3 and global_match_value_3 >= THRESHOLD:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}\n")
            return None, None, None, None

        if species_1 != species_3 and global_match_value_3 < THRESHOLD:
            log.write(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}\n")

    # Return the genus and species from the first match, and the externId
    log.write(f"✅ extern_id = {extern_id} at {target_position} accepted: {genus_1} {species_1} with {global_match_value_1}\n")
    return genus_1, species_1, extern_id, target_position

def process_xml(file_path, log):
    tree = ET.parse(file_path)
    root = tree.getroot()
    valid_entries = []

    for analyte in root.findall(".//Analyte"):
        genus, species, extern_id, target_position = process_analyte(analyte, log)
        if genus and species and extern_id and target_position:
            valid_entries.append((genus, species.capitalize(), extern_id, target_position))

    return valid_entries

def zip_and_move_folder(zip_ref: zipfile.ZipFile, src_path: str, dest_dir: str):
    for src_file_path in zip_ref.namelist():
        if not src_file_path.startswith(src_path):
            continue
        if src_file_path.endswith('/'):
            continue
        dest_file_path = os.path.join(dest_dir, src_file_path[len(src_path):])

        success = False
        for _ in range(3):
            # Write entry to disk
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            with open(dest_file_path, 'wb') as dest_file:
                dest_file.write(zip_ref.read(src_file_path))
            
            # Validate destination file
            if os.path.getsize(dest_file_path) > 0:
                success = True
                break
            print(f'Failed to write {dest_file_path}')

        # Handle severe fails
        if not success:
            raise Exception('Something is wrong with destination filesystem, panic')
        
        print(f"Success: {dest_file_path}")

def organize_files(valid_entries, zip_file_path, base_dest_dir, log):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for genus, species, extern_id, target_position in valid_entries:
            # Find the corresponding folder in the zip file
            folder_name = None
            for file in zip_ref.namelist():
                if f'/{extern_id}/0_{target_position}/' in file:
                    folder_name = file
                    if folder_name != f'{file.split(extern_id)[0]}{extern_id}/0_{target_position}/':
                        raise Exception(f'Illegal state for {file}') # There is no spectrum for the extern_id but there is some folder with the extern_id's name
                    break

            if not folder_name:
                log.write(f"🟥 Missing folder for accepted extern_id = {extern_id}\n")
                # raise Exception(f'Missing folder for {extern_id}')
            else:
                dest_dir = os.path.join(base_dest_dir, genus, species, extern_id, f'0_{target_position}')
                zip_and_move_folder(zip_ref, folder_name, dest_dir)

    print(f"Files have been organized into the MaldiMaranonDB structure")

def main():
    year = "2018"
    log_parsing_path = 'log-parsing.txt'

    with open(log_parsing_path, 'w', encoding='utf-8') as log:
        xml_dir = f"xml_to_parse_15798/{year}"

        # zip_file_path = f"Y:/{year}.zip"
        # base_dest_dir = "Z:/lschmidt/MaldiMaranonDB"

        zip_file_path = f"/export/usuarios01/lschmidt/{year}.zip"
        base_dest_dir = "/export/usuarios_ml4ds/lschmidt/MaldiMaranonDB"

        os.makedirs(base_dest_dir, exist_ok=True)

        for project in os.listdir(xml_dir):
            if project.endswith(".xml"):
                xml_file_path = os.path.join(xml_dir, project)
                log.write(f"Processing {xml_file_path}\n")
                print(f"Processing {xml_file_path}")
                valid_entries = process_xml(xml_file_path, log)
                organize_files(valid_entries, zip_file_path, base_dest_dir, log)

if __name__ == "__main__":
    main()
