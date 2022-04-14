import torch
from sklearn.metrics import confusion_matrix as skcm

def confusion_matrix(predictions, targets, labels):
    '''
    param: predictions: softmax scores 
    param: targets: ground truth prediction
    '''
    pred_classes = torch.argmax(predictions, dim=0)
    target_classes = torch.argmax(targets, dim=0)
    return skcm(pred_classes, target_classes)


def filter_incorrect(predictions, targets, wav_ids):
    pred_classes = torch.argmax(predictions, dim=1)
    target_classes = torch.argmax(targets, dim=1)
    incorrect_idxs = (pred_classes != target_classes).nonzero()
    true_targets = [target_classes[i] for i in incorrect_idxs]
    incorrect_predictions = [pred_classes[i] for i in incorrect_idxs]
    incorrect_wavs = [wav_ids[i] for i in incorrect_idxs]
    return incorrect_wavs, incorrect_predictions, true_targets   

# predictions = [
#     [0,0,1],
#     [0,1,0],
#     [0,1,0],
#     [1,0,0],
# ]
# targets = [
#     [0,1,0],
#     [1,0,0],
#     [0,1,0],
#     [0,1,0],
# ]


# argmax_p = [1, 1, 1]
# argmax_t = [1, 0, 1]

# labels = ['class_1','class_2','class_3']
# label_true_scores = [0 for i in range(len(labels))]
# label_fp_scores = [0 for i in range(len(labels))]
# label_fn_scores = [0 for i in range(len(labels))]

# matrix = [[] for i in range(len(labels))]
# for row_i, row in enumerate(predictions):
#     for value_i, prediction in enumerate(row):
#         # True positive
#         if prediction == 1 and targets[value_i][value_i] == 1:
#             label_true_scores[value_i] += 1
#         # False negative
#         elif prediction == 0 and targets[row_i][value_i] == 1:
#             label_fn_scores[value_i] += 1
#         # False positive
#         elif prediction == 1 and targets[row_i][value_i] == 0:
#             label_fp_scores[value_i] += 1
    
            
# for row in range(len(matrix)):
#     for value in range(len(labels)):
#         pass        
        

# Check if value is True positive
# Check if value is false negative 
# Check if value is false positive

gt_output = [
    [0, 1, 0],
    [0, 2, 0],
    [0, 0, 0],
]