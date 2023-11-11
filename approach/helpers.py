import numpy as np
import random

def read_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            row = line.split(',')
            row = [float(x) for x in row]
            data.append(row)
    return np.array(data)

def extract_last_column(matrix):
    last_column = matrix[:, -1].astype(float)
    return last_column[:, np.newaxis]

def extract_except_last_column(matrix):
    result = matrix[:, :-1].astype(float)
    return result

def select_zero_rows(data, num_zero_rows):
    zero_rows = []
    other_rows = []
    for row in data:
        if row[-1] == 0:
            zero_rows.append(row)
        else:
            other_rows.append(row)
    selected_zero_rows = random.sample(zero_rows, num_zero_rows)
    selected_data = np.concatenate((selected_zero_rows, other_rows))
    return selected_data

def convert_to_binary_label(y_pred, threshold):
    binary_labels = np.where(y_pred > threshold, 1, 0)
    return binary_labels

def cal_zeroOr1(data):
    data = np.array(data)

    last_column = data[:, -1]
    count_0 = np.count_nonzero(last_column == 0)
    count_1 = np.count_nonzero(last_column == 1)
    ratio_0 = count_0 / len(last_column)
    ratio_1 = count_1 / len(last_column)

    print("0 ratio:", ratio_0)
    print("1 ratio:", ratio_1)

def extract_loc_pred(filepath):
    loc_data = []
    pre_data = []
    y_true = []
    with open(filepath, 'r') as file:
        for line in file:
            if 'col18 :' in line:
                record_loc = True
                record_pre = False
                record_ytrue = False
                continue
            elif 'prediction :' in line:
                record_loc = False
                record_ytrue = False
                record_pre = True
                continue
            elif 'y_true: ' in line:
                record_loc = False
                record_ytrue = True
                record_pre = False
                continue

            if record_loc:
                loc_data.append(float(line.strip()))
            elif record_pre:
                pre_data.append(float(line.strip()))
            elif record_ytrue:
                y_true.append(float(line.strip()))

    return loc_data, pre_data, y_true

## index 0:y_true  1:LOC  2:predictions
def decArea(data):
    len_data = len(data)
    cumXs = np.cumsum(data[:, 1])  # x: LOC%
    cumYs = np.cumsum(data[:, 0])  # y: Bug%

    Xs = cumXs / cumXs[-1]
    Ys = cumYs / cumYs[-1]

    fix_subareas = np.ones(len_data)
    fix_subareas[0] = 0.5 * Ys[0] * Xs[0]
    fix_subareas[1:len_data] = 0.5 * (Ys[0:len_data-1] + Ys[1:len_data]) * np.abs(Xs[0:len_data-1] - Xs[1:len_data])

    area = np.sum(fix_subareas)
    return area

def sortData(data):
    # density = data[:, 0] / (data[:, 1] + 1)
    # data = np.column_stack((data, density))
    data = data[np.argsort(-data[:, 2])]

    y = data
    return y

def decPopt(data):
    data = sortData(data)
    data_mdl = data.copy()
    data_opt = data[np.lexsort((data[:, 1], -data[:, 3]))]
    data_wst = data[np.lexsort((-data[:, 1], data[:, 3]))]
    area_mdl = decArea(data_mdl)
    area_opt = decArea(data_opt)
    area_wst = decArea(data_wst)
    Popt = 1 - (area_opt - area_mdl) / (area_opt - area_wst)
    return Popt


def decR20E(data):
    data = sortData(data)
    length_data = len(data)
    cumXs = np.cumsum(data[:, 1])
    cumYs = np.cumsum(data[:, 0])
    Xs = cumXs / cumXs[length_data - 1]
    Ys = cumYs / cumYs[length_data - 1]
    pos = np.min(np.where(Xs >= 0.2))
    Recall_20_Effort = cumYs[pos] / cumYs[length_data - 1]
    return Recall_20_Effort

def cal_R20E_Popt(file_path):
    loc, pre, ytrue = extract_loc_pred(file_path)

    data = np.column_stack((ytrue, loc, pre))
    data = np.hstack((data, np.zeros((data.shape[0], 1))))
    data[:, 3] = data[:, 0] / (data[:, 1] + 1)
    popt = decPopt(data)
    R20E = decR20E(data)
    return R20E, popt

