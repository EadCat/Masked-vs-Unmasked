from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve


class Evaluator:
    def __init__(self, class_table:dict):
        self.classes = class_table
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.keys = list(self.classes.keys())
        self.values = list(self.classes.values())
        self.iter_counts = 0

    def __repr__(self):
        for k, v in zip(self.keys, self.values):
            print(f'number {k} : class {v}')

    def __str__(self):
        self.__repr__()

    def record(self, pred, gt):
        tn, fp, fn, tp = confusion_matrix(gt, pred, labels=self.keys).ravel()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.iter_counts += 1

    def precision(self):
        return (self.tp + 1e-10) / (self.tp + self.fp + 1e-10)

    def recall(self):
        return (self.tp + 1e-10) / (self.tp + self.fn + 1e-10)

    def f1_score(self):
        prec = self.precision()
        reca = self.recall()
        return (2 * (prec * reca)) / (prec + reca)

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

class CurvePlot:
    def __init__(self):
        pass