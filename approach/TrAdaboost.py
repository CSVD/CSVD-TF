import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from tqdm import tqdm

class Tradaboost(object):
    def __init__(self, N=None, base_estimator=None, threshold=None, score=roc_auc_score):
        self.N = N
        self.threshold = threshold
        self.base_estimator = base_estimator
        self.score = score
        self.estimators = []
        self.bata_T = np.zeros([1, self.N])

    def _calculate_weights(self, weights):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')

    def _calculate_error_rate(self, y_true, y_pred, weight):
        total = np.sum(weight)
        return np.sum(weight[:, 0] / total * np.abs(y_true - y_pred))

    def fit(self, source, target, source_label, target_label, early_stopping_rounds):

        best_round = 0
        source_shape = source.shape[0]
        target_shape = target.shape[0]
        trans_data = np.concatenate((source, target), axis=0)

        trans_label = np.concatenate((source_label, target_label), axis=0)
        weights_source = np.ones([source_shape, 1]) / source_shape
        weights_target = np.ones([target_shape, 1]) / target_shape
        weights = np.concatenate((weights_source, weights_target), axis=0)

        self.bata = 1 / (1 + np.sqrt(2 * np.log(source_shape / self.N)))
        self.bata_T = np.zeros([1, self.N])
        result_label = np.ones([source_shape + target_shape, self.N])

        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')

        score = 0
        flag = 0

        for i in tqdm(range(self.N)):
            P = self._calculate_weights(weights)
            P = P.reshape(-1)
            self.base_estimator.fit(trans_data, trans_label, P * 100)

            self.base_estimator.fit(trans_data, trans_label, sample_weight=P * 100)
            self.estimators.append(self.base_estimator)
            y_preds = self.base_estimator.predict_proba(trans_data)[:, 1]
            result_label[:, i] = y_preds

            y_target_pred = self.base_estimator.predict_proba(target)[:, 1]
            error_rate = self._calculate_error_rate(target_label, (y_target_pred > self.threshold).astype(int), \
                                                    weights[source_shape:source_shape + target_shape, :])

            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                N = i
                break

            self.bata_T[0, i] = error_rate / (1 - error_rate)

            for j in range(target_shape):
                weights[source_shape + j] = weights[source_shape + j] * \
                                            np.power(self.bata_T[0, i],
                                                     (-np.abs(result_label[source_shape + j, i] - target_label[j])))

            for j in range(source_shape):
                weights[j] = weights[j] * np.power(self.bata, np.abs(result_label[j, i] - source_label[j]))

            tp = self.score(target_label, y_target_pred)

            if tp > score:
                score = tp
                best_round = i
                flag = 0
            else:
                flag += 1
            if flag >= early_stopping_rounds:
                print('early stop!')
                break
        self.best_round = best_round
        self.best_score = score

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for estimator in self.estimators:
            y_pred = estimator.predict(X)
            predictions += y_pred
        predictions /= len(self.estimators)
        return predictions

