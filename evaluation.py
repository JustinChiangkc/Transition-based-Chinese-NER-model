import torch
from sklearn.metrics import f1_score

class F1_Calculator():
    def __init__(self):
        self.total_fn, self.total_tp, self.total_fp = 0, 0, 0
    def update(self, input):
        tp, fp, fn = input
        self.total_fn += fn
        self.total_tp += tp
        self.total_fp += fp
        return
    def __call__(self):
        return self._calculate_f1()
    def _calculate_f1(self):
        if (2*self.total_tp + self.total_fp + self.total_fn) == 0:
            f1 = 1
            return f1
        f1 = 2*self.total_tp / (2*self.total_tp + self.total_fp + self.total_fn)
        return f1

def cal_loss(pred_actions, gold_actions, loss_function):
    return loss_function(pred_actions, gold_actions)

def cal_act_acc(pred_actions, actions, pos_act, total_act):
    total_act += actions.shape[0]
    pred_label = torch.argmax(pred_actions, 1)
    pos = torch.sum(pred_label == actions)
    pos_act += pos
    return pos_act, total_act, float(pos_act) / total_act

def eval_seg(pred_seg, gold_seg):
    gold_seg = gold_seg[0]
    true_pos, false_pos, true_neg = 0, 0, 0
    for gold in pred_seg:
        if gold not in gold_seg:
            false_pos += 1
    for pred in pred_seg:
        if pred in gold_seg:
            gold_seg.remove(pred)
            true_pos += 1
        else:
            true_neg += 1
    return true_pos, false_pos, true_neg
def eval_pos(pred_pos, gold_pos):
    gold_pos = gold_pos[0]
    true_pos, false_pos, true_neg = 0, 0, 0
    for gold in pred_pos:
        if gold not in gold_pos:
            false_pos += 1
    for pred in pred_pos:
        if pred in gold_pos:
            gold_pos.remove(pred)
            true_pos += 1
        else:
            true_neg += 1
    return true_pos, false_pos, true_neg

def eval_ner(pred_ner, gold_ner):
    gold_ner = gold_ner[0]
    true_pos, false_pos, false_neg = 0, 0, 0
    for gold in gold_ner:
        if gold not in pred_ner:
            false_neg += 1
    for pred in pred_ner:
        if pred in gold_ner:
            gold_ner.remove(pred)
            true_pos += 1
        else:            
            false_pos += 1
    return true_pos, false_pos, false_neg


def eval_dep_labeled(pred_dep, gold_dep):
    pred_dep = [[x[0][1], x[1], x[2][1]] for x in pred_dep]
    gold_dep = [[x[1], x[2], x[3]] for x in gold_dep[0]]

    for i in range(len(gold_dep)):
        _, _, x = gold_dep[i]
        if x == 0:
            gold_dep[i][2] = 'ROOT' 
    true_pos, false_pos, true_neg = 0, 0, 0
    for gold in pred_dep:
        if gold not in gold_dep:
            false_pos += 1
    for pred in pred_dep:
        if pred in gold_dep:
            gold_dep.remove(pred)
            true_pos += 1
        else:
            true_neg += 1
    return true_pos, false_pos, true_neg

def eval_dep_unlabeled(pred_dep, gold_dep):
    pred_dep = [[x[0][1], x[2][1]] for x in pred_dep]
    gold_dep = [[x[1], x[3]] for x in gold_dep[0]]
    for i in range(len(gold_dep)):
        _, x = gold_dep[i]
        if x == 0:
            gold_dep[i][1] = 'ROOT' 
    true_pos, false_pos, true_neg = 0, 0, 0
    for gold in pred_dep:
        if gold not in gold_dep:
            false_pos += 1
    for pred in pred_dep:
        if pred in gold_dep:
            gold_dep.remove(pred)
            true_pos += 1
        else:
            true_neg += 1
    return true_pos, false_pos, true_neg
