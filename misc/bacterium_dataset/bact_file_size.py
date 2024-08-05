import re

def parse_file_sizes(file_path):
    total_size = 0
    size_pattern = re.compile(r'\d+(\.\d+)?[KMGT]')

    with open(file_path, 'r') as file:
        for line in file:
            if '.raw' in line:
                continue
            match = size_pattern.search(line)
            if match:
                size_str = match.group()
                size = float(size_str[:-1])
                unit = size_str[-1]

                # Convert size to bytes
                if unit == 'K':
                    size *= 1024
                elif unit == 'M':
                    size *= 1024**2
                elif unit == 'G':
                    size *= 1024**3
                elif unit == 'T':
                    size *= 1024**4

                total_size += size

    return total_size

def format_size(size):
    # Format size to a human-readable form
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

if __name__ == "__main__":
    file_path = 'misc/bact_file_list.txt'  # Path to the text file
    total_size = parse_file_sizes(file_path)
    formatted_size = format_size(total_size)
    print(f"Total size of files to be downloaded: {formatted_size}")
