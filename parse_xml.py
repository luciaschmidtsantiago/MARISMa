import os
import zipfile
import xml.etree.ElementTree as ET
import re

def process_analyte(analyte, log):
    extern_id = analyte.attrib.get("externId")
    target_position = analyte.attrib.get("targetPosition")
    msp_matches = analyte.findall(".//MspMatch") [:3]

    if not extern_id or not target_position or not msp_matches:
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Missing attributes\n")
        return None, None, None

    # Extract the referencePatternName and globalMatchValue from the first MspMatch
    ref_pattern_1 = msp_matches[0].attrib.get("referencePatternName")
    global_match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))

    if global_match_value_1 < 1.7:
        log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Low global_match_value ({global_match_value_1})\n")
        return None, None, None

    elif global_match_value_1 >= 1.7:
        genus_1, species_1 = ref_pattern_1.split()[:2]
        species_1 = re.sub(r'[^a-zA-Z]', ' ', species_1).split(" ")[0]
        
        # Check the second match
        ref_pattern_2 = msp_matches[1].attrib.get("referencePatternName")
        genus_2, species_2 = ref_pattern_2.split()[:2]
        species_2 = re.sub(r'[^a-zA-Z]', ' ', species_2).split(" ")[0]
        global_match_value_2 = float(msp_matches[1].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_2:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 2: {genus_2})\n")
            return None, None, None
        elif species_1 != species_2 and global_match_value_2 >= 1.7:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}\n")
            return None, None, None
        elif species_1 != species_2 and global_match_value_2 < 1.7:
            log.write(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 2: {species_2}): {global_match_value_2}\n")
        
        # Check the third match
        ref_pattern_3 = msp_matches[2].attrib.get("referencePatternName")
        genus_3, species_3 = ref_pattern_3.split()[:2]
        species_3 = re.sub(r'[^a-zA-Z]', ' ', species_3).split(" ")[0]
        global_match_value_3 = float(msp_matches[2].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_3:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Genus mismatch (1: {genus_1}, 3: {genus_3})\n")
            return None, None, None
        elif species_1 != species_3 and global_match_value_3 >= 1.7:
            log.write(f"❌ extern_id = {extern_id} at {target_position} denied: Species mismatch at high global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}\n")
            return None, None, None
        elif species_1 != species_3 and global_match_value_3 < 1.7:
            log.write(f"⚠️ extern_id = {extern_id} at {target_position} accepted: Species mismatch at low global_match_value (1: {species_1}, 3: {species_3}): {global_match_value_3}\n")

    # Return the genus and species from the first match, and the externId
    log.write(f"✅ extern_id = {extern_id} at {target_position} accepted: {genus_1} {species_1} with {global_match_value_1}\n")
    log.flush()
    return genus_1, species_1, extern_id

def process_xml(file_path, log):
    tree = ET.parse(file_path)
    root = tree.getroot()
    valid_entries = []

    for analyte in root.findall(".//Analyte"):
        genus, species, extern_id = process_analyte(analyte, log)
        if genus and species and extern_id:
            valid_entries.append((genus, species.capitalize(), extern_id))

    return valid_entries

def zip_and_move_folder(zip_ref, directory, extern_id, dest_dir, log):
    zip_folder_name = f"{extern_id}.zip"
    zip_folder_path = os.path.join(dest_dir, zip_folder_name)

    with zipfile.ZipFile(zip_folder_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in zip_ref.namelist():
            if file.startswith(directory):
                # Add file to the new zip file with the appropriate relative path
                zipf.writestr(os.path.relpath(file, directory), zip_ref.read(file))
        # Once we have gne through all the files, we can close the zip file
        zipf.close()

    print(f"Zipped folder: {zip_folder_path}")

def organize_files(valid_entries, zip_file_path, base_dest_dir, log):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for genus, species, extern_id in valid_entries:
            dest_dir = os.path.join(base_dest_dir, genus, species)
            os.makedirs(dest_dir, exist_ok=True)

            # Find the corresponding folder in the zip file
            folder_name = None
            for file in zip_ref.namelist():
                if extern_id in file:
                    folder_name = file.split(extern_id)[0] + extern_id + '/'
                    break

            if folder_name:
                zip_and_move_folder(zip_ref, folder_name, extern_id, dest_dir, log)

    print(f"Files have been organized into the MaldiMaranonDB structure")

def main():
    year = "2018"
    log_file_path = 'log-parsing.txt'

    with open(log_file_path, 'w', encoding='utf-8') as log:
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
