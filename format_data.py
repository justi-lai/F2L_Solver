import os
import re

FILE_NAMES = [
    "training.0",
    "training.1",
    "training.2",
    "training.3",
    "training.4",
    "training.5",
    "training.6",
    "training.7",
    "training.8",
    "training.9",
    "training.99"
]

for file_name in FILE_NAMES:
    file_path = os.path.join("dataset", file_name)
    if os.path.exists(file_path):
        # Read the file line by line and copy every second line to a new file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        new_file_name = re.sub(r'\.', '', file_name)  # Remove the period
        new_path = os.path.join("dataset", f"{new_file_name}_formatted.txt")
        with open(new_path, 'w') as new_file:
            for i, line in enumerate(lines):
                if i % 2 == 1:  # Copy every second line
                    new_file.write(line)
    else:
        print(f"File {file_name} does not exist.")