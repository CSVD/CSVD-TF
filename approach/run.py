from imblearn.over_sampling import SMOTE
from sklearn.model_selection import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

import warnings

import os
import datetime

import helpers
from TrAdaboost import Tradaboost

run_times = 200
testSize = 0.7

Trate = 0.25
params = {'booster': 'gbtree',
          'eta': 0.1,
          'max_depth': 20,
          'max_delta_step': 0,
          'subsample': 1,
          'colsample_bytree': 1,
          'base_score': Trate,
          'objective': 'binary:logistic',
          'lambda': 5,
          'alpha': 8,
          'n_estimators': 500,
          'n_jobs': -1
          }


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    sourcelist = [
        'FFmpeg.txt',
        'LibPNG.txt',
        'LibTIFF.txt',
        'VLC.txt'
        ]
    targetlist = [
        'FFmpeg.txt',
        'LibPNG.txt',
        'LibTIFF.txt',
        'VLC.txt'
        ]

    current_time = datetime.datetime.now()

    time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    father_folder = time_string
    os.mkdir(father_folder)


    for id in range(0, 10):
        folder_name = father_folder + '/CSVD-TF_testId-' + str(id)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for sourcename in sourcelist:
            for targetname in targetlist:
                if sourcename == targetname:
                    continue

                source_path = '../data/' + sourcename
                target_path = '../data/' + targetname
                Source = helpers.read_file(source_path)
                Target = helpers.read_file(target_path)

                Xsource = helpers.extract_except_last_column(Source)
                ysource = helpers.extract_last_column(Source)
                Xtarget = helpers.extract_except_last_column(Target)
                ytarget = helpers.extract_last_column(Target)

                Xsource_expert = Xsource[:, :39]
                Xsource_semantic = Xsource[:, 39:]
                Xtarget_expert = Xtarget[:, :39]
                Xtarget_semantic = Xtarget[:, 39:]

                smote_expert = SMOTE()
                Xsource_expert, ysource_expert = smote_expert.fit_resample(Xsource_expert, ysource)
                smote_semantic = SMOTE()
                Xsource_semantic, ysource_semantic = smote_semantic.fit_resample(Xsource_semantic, ysource)

                train_B_1_valid, train_B_1_test, train_B_1_valid_y, train_B_1_test_y = train_test_split(Xtarget, ytarget,
                                                                                                test_size=testSize, stratify=ytarget)

                train_B_1_valid_expert = train_B_1_valid[:, :39]
                train_B_1_valid_semantic = train_B_1_valid[:, 39:]
                train_B_1_test_expert = train_B_1_test[:, :39]
                train_B_1_test_semantic = train_B_1_test[:, 39:]

                train_B_1_valid_expert, train_B_1_valid_y_expert = smote_expert.fit_resample(train_B_1_valid_expert, train_B_1_valid_y)
                train_B_1_valid_semantic, train_B_1_valid_y_semantic = smote_expert.fit_resample(train_B_1_valid_semantic, train_B_1_valid_y)

                train_B_1_valid_y = np.squeeze(train_B_1_valid_y)
                train_B_1_test_y = np.squeeze(train_B_1_test_y)

                print(sourcename + '->' + targetname)

                print('start training the expert metric-based model ...')
                ## expert metric-based model training
                clf_expert = Tradaboost(N=run_times, base_estimator=xgb.XGBClassifier(**params),
                           threshold=0.92975, score=roc_auc_score)

                clf_expert.fit(Xsource_expert, train_B_1_valid_expert, ysource_expert, train_B_1_valid_y_expert, 50)

                ## expert metric-based model prediction
                y_pred_B_test_expert = clf_expert.predict(train_B_1_test_expert)

                print('start training the semantic metric-based model ...')
                ## semantic metric-based model training
                clf_semantic = Tradaboost(N=run_times, base_estimator=xgb.XGBClassifier(**params),
                                          threshold=0.92975, score=roc_auc_score)

                clf_semantic.fit(Xsource_semantic, train_B_1_valid_semantic, ysource_semantic, train_B_1_valid_y_semantic, 50)

                ## semantic metric-based model prediction
                y_pred_B_test_semantic = clf_semantic.predict(train_B_1_test_semantic)

                ## assignment the weights of expert metric-based model and semantic metric-based model
                expert_weight = 0.4
                semantic_weight = 0.6

                ## calculate the final prediction
                predictions = y_pred_B_test_expert * expert_weight + y_pred_B_test_semantic * semantic_weight

                loc = [row[18] for row in train_B_1_test_expert]
                pred_label = predictions.round()

                filename = sourcename.split('.')[0] + "_" + targetname
                res_file = './' + folder_name + '/' + 'res_' + filename
                with open(res_file, 'w') as file:
                    file.write("col18 :\n")
                    for item in loc:
                        file.write(str(item) + "\n")

                    file.write("prediction :\n")
                    for item in pred_label:
                        file.write(str(item) + "\n")

                    file.write("y_true: \n")
                    for item in train_B_1_test_y:
                        file.write(str(item) + "\n")

                auc = roc_auc_score(train_B_1_test_y, predictions)
                R20E, popt = helpers.cal_R20E_Popt(res_file)

                file_path = './' + folder_name + '/' + filename
                with open(file_path, 'w') as f:
                    content = 'AUC:' + str(auc) + '\n'
                    content += '==============================================================\n'
                    content += 'R20E:' + str(R20E) + '\n'
                    content += 'popt:' + str(popt)
                    f.write(content)
                    f.close()