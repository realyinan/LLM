from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


class ClassEvaluator(object):
    def __init__(self):
        self.goldens = []
        self.predictions = []

    def add_batch(self, pred_batch: List[list], gold_batch: List[list]):
        """
        添加一个batch中的prediction和gold列表，用于后续统一计算。
        Args:
            pred_batch (list): 模型预测标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
            gold_batch (list): 真实标签标签列表, e.g. ->  [['体', '育'], ['财', '经'], ...]
        """
        assert len(pred_batch) == len(gold_batch)

        if type(gold_batch[0]) in [list, tuple]:
            pred_batch = ["".join([str(e) for e in ele]) for ele in pred_batch]
            gold_batch = ["".join([str(e) for e in ele]) for ele in gold_batch]

        self.goldens.extend(gold_batch)
        self.predictions.extend(pred_batch)

    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1。
        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。
        Returns:
            dict -> {
                'accuracy': 准确率,
                'precision': 精准率,
                'recall': 召回率,
                'f1': f1值,
                'class_metrics': {
                    '0': {
                            'precision': 该类别下的precision,
                            'recall': 该类别下的recall,
                            'f1': 该类别下的f1
                        },
                    ...
                }
            }
        """
        # print("goldens-->", self.goldens)
        # print("predictions-->", self.predictions)
        classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {}

        res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)  # 构建全局指标
        res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)

        try:
            conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))  # 混淆矩阵内部也会对每个类别排序
            assert conf_matrix.shape[0] == len(classes)

            for i in range(conf_matrix.shape[0]):
                precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
                recall = 0 if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
                f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                class_metrics[classes[i]] = {
                    "precision": round(precision, round_num),
                    "recall": round(recall, round_num),
                    "f1": round(f1, round_num)
                }
            res["class_metrics"] = class_metrics

        except Exception as e:
            print(f'[Warning] Something wrong when calculate class_metrics: {e}')
            print(f'-> goldens: {set(self.goldens)}')
            print(f'-> predictions: {set(self.predictions)}')
            print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
            res['class_metrics'] = {}
        return res

    def reset(self):
        """
        重置积累的数值。
        """
        self.goldens = []
        self.predictions = []


if __name__ == '__main__':
    from rich import print

    metric = ClassEvaluator()
    metric.add_batch(
        [['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']],
        [['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']],
    )
    print(metric.compute())
