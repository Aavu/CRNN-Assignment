import numpy as np
import torch
from sklearn.metrics import r2_score, accuracy_score

"""
Contains standard utility functions for training and testing evaluations
"""


def eval_regression(target, pred):
    """
    Calculates the standard regression evaluation  metrics
    Args:
        target:     (N x 1) torch Float tensor, actual ground truth
        pred:       (N x 1) torch Float tensor, predicted values from the regression model
    Returns:
        r_sq:       float, average r-squared metric
        accu:       float, average accuracy (between 0. to 1.)
    """
    r_sq = r2_score(target, pred)
    accu = accuracy_score(target * 10, np.round(pred, 1) * 10, normalize=True)
    return r_sq, accu


def eval_model(model, criterion, data, metric, extra_outs=False):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
        extra_outs:     bool, returns the target and predicted values if true
    """
    # put the model in eval mode
    model.eval()
    # intialize variables
    num_batches = len(data)
    pred = np.array([])
    target = np.array([])
    loss_avg = 0
    # iterate over batches for validation
    for batch_idx in range(num_batches):
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]

        # model.init_hidden(pitch_tensor.size()[0])
        if torch.cuda.is_available():
            pitch_tensor = pitch_tensor.cuda()
            score_tensor = score_tensor.cuda()
        model_output = model(pitch_tensor)
        loss = criterion(model_output, score_tensor)
        loss_avg += loss.item()

        # concatenate target and pred for computing validation metrics
        pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
    r_sq, accu = eval_regression(target, pred)
    loss_avg /= num_batches
    if extra_outs:
        return loss_avg, r_sq, accu, pred, target
    else:
        return loss_avg, r_sq, accu
