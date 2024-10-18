import pyzipper

zip_file_path = '/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/XML_2018a2022.zip'

try:
    with pyzipper.AESZipFile(zip_file_path) as zip_ref:
        # List the contents of the zip file
        print("Contents of the ZIP file:")
        print(zip_ref.namelist())
except Exception as e:
    print(f"An error occurred: {e}")
