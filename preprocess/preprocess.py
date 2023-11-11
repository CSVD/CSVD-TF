import os
import re
import csv
import numpy as np
import bert_whitening
import pandas as pd
import pickle


## Check if the file is empty
def judge_null(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            with open(file_path, 'w') as f:
                f.write("Nothing")
            print("Empty file filled", file_path, "is NULL")

# Write code to txt file
def CodeToTxt(file_path):
    # Define source code folder path
    folder = file_path
    # Obtain the paths of all C source files
    c_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder) for f in filenames if
               os.path.splitext(f)[1] == '.c']

    # Write the code for each source file to a CSV file
    csv_path = file_path + '_code.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for c_file in c_files:
            with open(c_file, 'r', encoding='gb2312', errors='ignore') as c_f:

                code = c_f.read()
                code = re.sub(r'\/\*[\s\S]*?\*\/|\/\/.*', '', code)
                code = ''.join(line.lstrip() for line in code.splitlines())
                curly_braces = 0
                new_code = ''
                for char in code:
                    if char == '{':
                        curly_braces += 1
                    elif char == '}':
                        curly_braces -= 1
                        if curly_braces < 0:
                            curly_braces = 0
                            continue
                    if curly_braces > 0 or char not in {' ', '\t'}:
                        new_code += char

                new_code = re.sub(r'^[^a-zA-Z]*', '', new_code)

                if not new_code:
                    continue

                writer.writerow([new_code])

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        output_txt = 'output_' + file_path + '.txt'
        with open(output_txt, "w") as g:
            for row in reader:
                row_str = ",".join(row) + "\n"
                g.write(row_str)


# Obtain file label
def get_label(dir_path):
    cve_files = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".c"):
            if re.search("[Cc][Vv][Ee]-\d+", filename):  # judge vulnerabilities
                cve_files.append(1)
            else:
                cve_files.append(0)
    cve_files_array = np.array(cve_files)
    label_path = dir_path + '_label.npy'
    np.save(label_path, cve_files_array)

# Concatenation of Expert metrics and semantic metrics
def data_Concatenation(expert_file_path, semantic_file_path, Concatenation_file_path):
    # load the expert metrics
    df = pd.read_csv(expert_file_path, header=None, delimiter=",")

    # load the semantic metrics
    with open(semantic_file_path, "rb") as f:
        data = pickle.load(f)

    # Concatenation
    result = []
    for i in range(len(data)):
        row = np.concatenate([df.iloc[i].values, data[i]])
        result.append(row)

    np.savetxt(Concatenation_file_path, result, delimiter=",")


def addlabel(Concatenation_file_path,label_path):
    with open(Concatenation_file_path, 'r') as f:
        lines = f.readlines()

    labels = np.load(label_path)
    new_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        data = line.split(',')
        data.append(str(labels[i]))
        new_lines.append(','.join(data))

def main():
    dir_path = 'VLC'
    judge_null(dir_path)
    CodeToTxt(dir_path)
    get_label(dir_path)
    bert_whitening.main()
    expert_file_path = 'expert_VLC.txt'
    semantic_file_path = 'semantic_VLC.pkl'
    Concatenation_file_path = 'Concatenation_VLC.txt'
    data_Concatenation(expert_file_path, semantic_file_path, Concatenation_file_path)
    label_path='VLC_label.npy'
    addlabel(Concatenation_file_path,label_path)




