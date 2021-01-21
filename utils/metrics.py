import numpy as np

def mse(y_pred, y_true):
    '''input:
    y_pred: (k, 1)
    y_true: (k, 1)'''
    y_pred = y_pred.reshape(y_true.shape)
    return np.sum((y_pred - y_true)**2) / y_true.shape[0]
    
def mape(y_pred, y_true):
    '''input:
    y_pred: (k, 1)
    y_true: (k, 1)'''
    y_pred = y_pred.reshape(y_true.shape)
    return np.sum(np.abs((y_pred - y_true) / y_true)) / y_true.shape[0]

def contingency_table(y_pred, y_true):
    '''input:
    y_pred: (k, 1)
    y_true: (k, 1)'''
    y_pred = y_pred.reshape(y_true.shape)
    TP = len(y_pred[(y_pred==1) & (y_true ==1)])
    TN = len(y_pred[(y_pred==0) & (y_true ==0)])
    FP = len(y_pred[(y_pred==1) & (y_true ==0)])
    FN = len(y_pred[(y_pred==0) & (y_true ==1)])
    return TP, TN, FP, FN
    
def accuracy(y_pred, y_true):
    TP, TN, FP, FN = contingency_table(y_pred, y_true)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_pred, y_true):
    TP, TN, FP, FN = contingency_table(y_pred, y_true)
    return TP / (TP + FP)

def recall(y_pred, y_true):
    TP, TN, FP, FN = contingency_table(y_pred, y_true)
    return TP / (TP + FN)

def f1_score(y_pred, y_true):
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return safe_divide(2*prec*rec , (prec + rec))

def safe_divide(a, b):
    if b == 0:
        b = 1e-4
    return a / b

def to_binary(y, i):
        '''input: i = the class label to consider as 1
        turns multiclass y into binary y'''
        y = y + 2 # in case i in [0,1]
        y[y != i+2] = 0
        y[y == i+2] = 1
        return y

def multiclass_accuracy(y_pred, y_true):
    y_pred = y_pred.reshape(y_true.shape)
    return len(y_pred[y_pred == y_true]) / len(y_pred)

def multiclass_precision(y_pred, y_true):
    y_pred = y_pred.reshape(y_true.shape)
    tables = {}
    for cl in np.unique(y_true):
        y_pred_binary = to_binary(y_pred, cl)
        y_true_binary = to_binary(y_true, cl)
        TP, TN, FP, FN = contingency_table(y_pred_binary, y_true_binary)
        tables[cl] = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    micro_avg = safe_divide(sum([tables[i]['TP'] for i in tables]) , sum([tables[i]['TP'] + tables[i]['FP'] for i in tables]))
    macro_avg = np.mean([safe_divide(tables[i]['TP'] , (tables[i]['TP'] + tables[i]['FP'])) for i in tables])
    return {'micro_avg':micro_avg, 'macro_avg':macro_avg}

def multiclass_recall(y_pred, y_true):
    y_pred = y_pred.reshape(y_true.shape)
    tables = {}
    for cl in np.unique(y_true):
        y_pred_binary = to_binary(y_pred, cl)
        y_true_binary = to_binary(y_true, cl)
        TP, TN, FP, FN = contingency_table(y_pred_binary, y_true_binary)
        tables[cl] = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    micro_avg = safe_divide(sum([tables[i]['TP'] for i in tables]) , sum([tables[i]['TP'] + tables[i]['FN'] for i in tables]))
    macro_avg = np.mean([safe_divide(tables[i]['TP'] , (tables[i]['TP'] + tables[i]['FN'])) for i in tables])
    return {'micro_avg':micro_avg, 'macro_avg':macro_avg}
   
def multiclass_f1_score(y_pred, y_true):
    prec = multiclass_precision(y_pred, y_true) 
    rec = multiclass_recall(y_pred, y_true)
    micro_avg = safe_divide(2*prec['micro_avg']*rec['micro_avg'] , (prec['micro_avg'] + rec['micro_avg']))
    macro_avg = safe_divide(2*prec['macro_avg']*rec['macro_avg'] , (prec['macro_avg'] + rec['macro_avg']))
    return {'micro_avg':micro_avg, 'macro_avg':macro_avg}

def classification_report(y_pred, y_true):
    if len(np.unique(y_true)) == 2:
        return {'accuracy':accuracy(y_pred, y_true),
                'precision':precision(y_pred, y_true),
                'recall':recall(y_pred, y_true),
                'f1_score':f1_score(y_pred, y_true)}
    else:
        return {'accuracy':multiclass_accuracy(y_pred, y_true),
                'precision':multiclass_precision(y_pred, y_true),
                'recall':multiclass_recall(y_pred, y_true),
                'f1_score':multiclass_f1_score(y_pred, y_true)}