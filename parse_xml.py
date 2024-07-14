import os
import zipfile
import xml.etree.ElementTree as ET
import shutil

def process_analyte(analyte):
    extern_id = analyte.attrib.get("externId")
    target_position = analyte.attrib.get("targetPosition")
    msp_matches = analyte.findall(".//MspMatch") [:3]

    if not extern_id or not target_position or not msp_matches:
        print(f"Skipping analyte. Missing attributes.")
        return None, None, None

    # Extract the referencePatternName and globalMatchValue from the first MspMatch
    ref_pattern_1 = msp_matches[0].attrib.get("referencePatternName")
    global_match_value_1 = float(msp_matches[0].attrib.get("globalMatchValue", "0"))

    if global_match_value_1 < 1.7:
        print(f"Skipping analyte {extern_id} at {target_position}. Low global_match_value ({global_match_value_1}).")
        return None, None, None

    elif global_match_value_1 >= 1.7:
        genus_1, species_1 = ref_pattern_1.split()[:2]
        
        # Check the second match
        ref_pattern_2 = msp_matches[1].attrib.get("referencePatternName")
        genus_2, species_2 = ref_pattern_2.split()[:2]
        global_match_value_2 = float(msp_matches[1].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_2:
            print(f"Skipping analyte {extern_id} at {target_position}. Genus mismatch ({genus_1, genus_2}).")
            return None, None, None
        elif species_1 != species_2 and global_match_value_2 >= 1.7:
            print(f"Skipping analyte {extern_id} at {target_position}. Species mismatch with high global_match_value ({species_1, species_2, global_match_value_2}).")
            return None, None, None
        elif species_1 != species_2 and global_match_value_2 < 1.7:
            print(f"Accepting analyte {extern_id} at {target_position}. Species mismatch with low global_match_value ({global_match_value_2}).\n Genus: {genus_1}\n  Species 1: {species_1}\n  Species 2: {species_2}")
        
        # Check the third match
        ref_pattern_3 = msp_matches[2].attrib.get("referencePatternName")
        genus_3, species_3 = ref_pattern_3.split()[:2]
        global_match_value_3 = float(msp_matches[2].attrib.get("globalMatchValue", "0"))

        if genus_1 != genus_3:
            print(f"Skipping analyte {extern_id} at {target_position}. Genus mismatch ({genus_1, genus_3}).")
            return None, None, None
        elif species_1 != species_3 and global_match_value_3 >= 1.7:
            print(f"Skipping analyte {extern_id} at {target_position}. Species mismatch with high global_match_value ({species_1, species_3, global_match_value_3}).")
            return None, None, None
        elif species_1 != species_3 and global_match_value_3 < 1.7:
            print(f"Accepting analyte {extern_id} at {target_position}. Species mismatch with low global_match_value ({global_match_value_3}).\n Genus: {genus_1}\n  Species 1: {species_1}\n  Species 3: {species_3}")

    # Return the genus and species from the first match, and the externId
    print('Accepted analyte:', genus_1, species_1, extern_id)
    return genus_1, species_1, extern_id

def process_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    valid_entries = []

    for analyte in root.findall(".//Analyte"):
        genus, species, extern_id = process_analyte(analyte)
        if genus and species and extern_id:
            valid_entries.append((genus, species.capitalize(), extern_id))

    return valid_entries

def zip_and_move_folder(zip_ref, folder_name, extern_id, dest_dir):
    zip_folder_name = f"{extern_id}.zip"
    zip_folder_path = os.path.join(dest_dir, zip_folder_name)

    with zipfile.ZipFile(zip_folder_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in zip_ref.namelist():
            if file.startswith(folder_name):
                # Add file to the new zip file with the appropriate relative path
                zipf.writestr(os.path.relpath(file, folder_name), zip_ref.read(file))

    print(f"Zipped folder: {zip_folder_path}")

def organize_files(valid_entries, zip_file_path, base_dest_dir):
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
                zip_and_move_folder(zip_ref, folder_name, extern_id, dest_dir)

    print(f"Files have been organized into the MaldiMaranonDB structure")

def main():
    year = "2018"

    xml_dir = f"xml_to_parse_15798/{year}"
    zip_file_path = f"/export/usuarios01/lschmidt/{year}.zip"
    base_dest_dir = "/export/usuarios_ml4ds/lschmidt/MaldiMaranonDB"

    os.makedirs(base_dest_dir, exist_ok=True)

    for project in os.listdir(xml_dir):
        if project.endswith(".xml"):
            xml_file_path = os.path.join(xml_dir, project)
            print(f"Processing {xml_file_path}")
            valid_entries = process_xml(xml_file_path)
            organize_files(valid_entries, zip_file_path, base_dest_dir)

if __name__ == "__main__":
    main()
