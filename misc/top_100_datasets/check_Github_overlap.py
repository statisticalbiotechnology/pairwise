import subprocess
import requests

# Define the command to be run
command = "grep -o 'PXD[0-9]\\+' misc/top_100_pride.txt | sort | uniq"

# Run the command and capture the output
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
)
stdout, stderr = process.communicate()

# Convert the byte-string output to a string and split by newlines to get a list
px_identifiers = stdout.decode("utf-8").strip().split("\n")

# Base URL for GitHub API to access the contents of the repository
base_url = "https://api.github.com/repos/bigbio/proteomics-sample-metadata/contents/annotated-projects"

# Fetch the list of directories only once
response = requests.get(base_url)
if response.status_code == 200:
    # Extract directory names
    dir_names = [item["name"] for item in response.json() if item["type"] == "dir"]

    # Find and store overlapping identifiers
    found_identifiers = [px_id for px_id in px_identifiers if px_id in dir_names]
else:
    print("Failed to fetch the GitHub directories.")
    raise SystemExit(1)

# Print out found identifiers and their count
print(f"Found {len(found_identifiers)} identifiers overlapping with GitHub folders.")
if found_identifiers:
    print("Overlapping identifiers:")
    for found_id in found_identifiers:
        print(found_id)
else:
    print("No overlapping identifiers found.")
