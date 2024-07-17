import os
import argparse
from ftplib import FTP

# Define the FTP server and directory
ftp_server = "ftp.pride.ebi.ac.uk"
# ftp_directory = "/pride/data/archive/2018/06/PXD010000/"
ftp_directory = "/pride/data/archive/2022/09/PXD010613/"

# Define the local directory to save the files
# local_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010000"
local_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010613"

# Create the local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

def list_ftp_files(ftp):
    # List the files in the directory, excluding raw files
    files = ftp.nlst()
    non_raw_files = [file for file in files if not file.endswith('.raw')]
    return non_raw_files

def download_ftp_files(files):
    # Connect to the FTP server
    with FTP(ftp_server) as ftp:
        ftp.login()
        ftp.cwd(ftp_directory)
        
        # Download each file
        for file_name in files:
            local_file_path = os.path.join(local_directory, file_name)
            with open(local_file_path, 'wb') as local_file:
                ftp.retrbinary(f"RETR {file_name}", local_file.write)
                print(f"Downloaded: {file_name}")

def main():
    parser = argparse.ArgumentParser(description='Download files from an FTP server.')
    parser.add_argument('-y', '--yes', action='store_true', help='Automatic yes to prompts')
    args = parser.parse_args()
    
    # Connect to the FTP server to list files
    with FTP(ftp_server) as ftp:
        ftp.login()
        ftp.cwd(ftp_directory)
        files = list_ftp_files(ftp)
    
    # Print the list of files to be downloaded
    print("The following files will be downloaded:")
    for file_name in files:
        print(file_name)
    
    # Ask for user confirmation if -y is not passed
    if not args.yes:
        confirm = input("Do you want to proceed with the download? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Download cancelled.")
            return
    
    # Proceed with the download
    download_ftp_files(files)

if __name__ == "__main__":
    main()
