import os

directory = '/Users/soniaqi/Documents/CSCI467/csci467-project/zh2en/BWB_dataset/train'

output_dir = '/Users/soniaqi/Documents/CSCI467/csci467-project/zh2en/'

'''
def collect_files_by_suffix(root_dir, suffix):
    files = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                prefix = rel_path[:-len(suffix)]  # remove suffix like '.enu'
                files[prefix] = full_path
    return files

def combine_matched_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    enu_files = collect_files_by_suffix(input_dir, 'mt_re.txt')
    chs_files = collect_files_by_suffix(input_dir, 'chs_re.txt')

    common_prefixes = sorted(set(enu_files) & set(chs_files))

    enu_output_path = os.path.join(output_dir, 'testENU.txt')
    chs_output_path = os.path.join(output_dir, 'testCHS.txt')

    with open(enu_output_path, 'w', encoding='utf-8') as enu_out, \
         open(chs_output_path, 'w', encoding='utf-8') as chs_out:

        for prefix in common_prefixes:
            with open(enu_files[prefix], 'r', encoding='utf-8') as f:
                enu_out.write(f.read())
                enu_out.write('\n')  # optional
            with open(chs_files[prefix], 'r', encoding='utf-8') as f:
                chs_out.write(f.read())
                chs_out.write('\n')  # optional

    print(f"Combined {len(common_prefixes)} matched file pairs.")
    unmatched_enu = set(enu_files) - set(chs_files)
    unmatched_chs = set(chs_files) - set(enu_files)
    if unmatched_enu or unmatched_chs:
        print(f"Unmatched .enu files: {sorted(unmatched_enu)}")
        print(f"Unmatched .chs files: {sorted(unmatched_chs)}")

# Example usage
input_dir = directory
combine_matched_files(input_dir, output_dir)


def combine_matched_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Collect .enu and .chs files by prefix
    enu_files = {}
    chs_files = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.enu'):
            prefix = filename[:-4]
            enu_files[prefix] = os.path.join(input_dir, filename)
        elif filename.endswith('.chs'):
            prefix = filename[:-4]
            chs_files[prefix] = os.path.join(input_dir, filename)

    # Find common prefixes and sort them
    common_prefixes = sorted(set(enu_files) & set(chs_files))

    # Define output paths
    enu_output_path = os.path.join(output_dir, 'trainENU.txt')
    chs_output_path = os.path.join(output_dir, 'trainCHS.txt')

    # Write combined files
    with open(enu_output_path, 'w', encoding='utf-8') as enu_out, \
         open(chs_output_path, 'w', encoding='utf-8') as chs_out:

        for prefix in common_prefixes:
            with open(enu_files[prefix], 'r', encoding='utf-8') as ef:
                enu_out.write(ef.read())
                enu_out.write('\n')  # optional: separate files

            with open(chs_files[prefix], 'r', encoding='utf-8') as cf:
                chs_out.write(cf.read())
                chs_out.write('\n')

    print(f"Combined {len(common_prefixes)} matched file pairs.")

    # Report unmatched files
    unmatched_enu = set(enu_files) - set(chs_files)
    unmatched_chs = set(chs_files) - set(enu_files)
    if unmatched_enu:
        print(f"Unmatched .enu files: {sorted(unmatched_enu)}")
    if unmatched_chs:
        print(f"Unmatched .chs files: {sorted(unmatched_chs)}")

# Example usage
input_dir = directory
combine_matched_files(input_dir, output_dir)

# Read the contents of the file
with open("trainENU.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Replace <sep> with newline
content = content.replace("<sep>", "\n")

# Write the modified content back to the file (or to a new file)
with open("trainENU.txt", "w", encoding="utf-8") as file:
    file.write(content)

from itertools import zip_longest

with open("trainENU.txt", "r", encoding="utf-8") as en, open("trainCHS.txt", "r", encoding="utf-8") as zh:
    for i, (en_line, zh_line) in enumerate(zip(en, zh), 1):
        if i % 100 == 0:
            print(f"Line {i}:\nEN: {en_line.strip()}\nZH: {zh_line.strip()}\n{'-'*40}")
        
def truncate_file(input_path, output_path, max_lines):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            if i >= max_lines:
                break
            outfile.write(line)

# Truncate to first 2,181,300 lines
truncate_file("trainENU.txt", "trucENU.txt", 50000)
truncate_file("trainCHS.txt", "trucCHS.txt", 50000)

def read_zh_data(fname):
    with open(fname, encoding="utf-8") as f:
        lines = [line.rstrip('\n') for line in f]
    return lines
src = read_zh_data("trucCHS.txt")
tgt = read_zh_data("trucENU.txt")
print("lensrc: ",len(src))
print("lentgt: ",len(tgt))
'''
with open("testENU.txt", "r", encoding="utf-8") as f:
    num_non_empty_lines = sum(1 for line in f if line.strip())
print(f"Number of non-empty lines: {num_non_empty_lines}")
with open("testCHS.txt", "r", encoding="utf-8") as f:
    num_non_empty_lines = sum(1 for line in f if line.strip())
print(f"Number of non-empty lines: {num_non_empty_lines}")