import math
import time

import torch

import eval_utils


def augment_data(data):
    """
    Augments the data using pitch shifting
    Args:
        data: batched data
    """
    num_batches = len(data)
    aug_data = [None] * num_batches
    # create augmented data
    for batch_idx in range(num_batches):
        mini_batch_size, seq_len = data[batch_idx]['pitch_tensor'].size()
        pitch_shift = ((torch.rand(mini_batch_size, 1) * 4) - 2) / 72.0  # since we normalize between 36 to 108 MIDI Note
        pitch_shift = pitch_shift.expand(mini_batch_size, seq_len)
        pitch_tensor = data[batch_idx]['pitch_tensor'].clone()
        pitch_tensor[pitch_tensor != 0] = pitch_tensor[pitch_tensor != 0] + pitch_shift[pitch_tensor != 0]
        new_data = {}
        new_data['pitch_tensor'] = pitch_tensor
        new_data['score_tensor'] = data[batch_idx]['score_tensor'].clone()
        aug_data[batch_idx] = new_data
    # combine with orignal data
    aug_data = data + aug_data
    return aug_data


def train(model, criterion, optimizer, data, metric):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
    Returns:            (1,) torch Tensor, average MSE loss for all batches
    """
    # Put the model in training mode
    model.train()
    # Initializations
    num_batches = len(data)
    loss_avg = 0
    # iterate over batches for training
    for batch_idx in range(num_batches):
        # clear gradients and loss
        model.zero_grad()
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        optimizer.zero_grad()
        # model.init_hidden(pitch_tensor.size()[0])
        if torch.cuda.is_available():
            pitch_tensor = pitch_tensor.cuda()
            score_tensor = score_tensor.cuda()
        outputs = model(pitch_tensor)
        loss = criterion(outputs, score_tensor)
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()

    loss_avg /= num_batches
    return loss_avg


# define training and validate method
def train_and_validate(model, criterion, optimizer, train_data, val_data, metric):
    """
    Defines the training and validation cycle for the input batched data for the conv model
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        train_data:     list, batched training data
        val_data:       list, batched validation data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # train the network
    train(model, criterion, optimizer, train_data, metric)
    # evaluate the network on train data
    train_loss_avg, train_r_sq, train_accu = eval_utils.eval_model(model, criterion, train_data, metric)
    # evaluate the network on validation data
    val_loss_avg, val_r_sq, val_accu = eval_utils.eval_model(model, criterion, val_data, metric)
    # return values
    return train_loss_avg, train_r_sq, train_accu, val_loss_avg, val_r_sq, val_accu


def save(filename, perf_model, log_parameters=None):
    """
    Saves the saved model
    Args:
        filename:       name of the file 
        model:          torch.nn model 
        log_parameters: dict, contaning the log parameters
    """
    save_filename = 'saved/' + filename + '_Reg.pt'
    torch.save(perf_model.state_dict(), save_filename)
    if log_parameters is not None:
        log_filename = 'runs/' + filename + '_Log.txt'
        f = open(log_filename, 'w')
        f.write(str(log_parameters))
        f.close()
    print('Saved as %s' % save_filename)


def time_since(since):
    """
    Returns the time elapsed between now and 'since'
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def adjust_learning_rate(optimizer, epoch, adjust_every):
    """
    Adjusts the learning rate of the optimizer based on the epoch
    Args:
       optimizer:      object, of torch.optim class 
       epoch:          int, epoch number
       adjust_every:   int, number of epochs after which adjustment is to done
    """
    gamma = 0.5
    temp = 0
    if epoch % adjust_every == 0:
        for param_group in optimizer.param_groups:
            temp = param_group['lr'] * gamma
            param_group['lr'] = temp
        print("learning rate changed to", temp)


def log_init():
    """
    Initializes the log element
    """
    log_parameters = {
        'x': [],
        'loss_train': [],
        'r_sq_train': [],
        'acc_train': [],
        'loss_val': [],
        'r_sq_val': [],
        'acc_val': [],
    }
    return log_parameters


def log_epoch_stats(
        log_parameters,
        epoch_index,
        mean_loss_train,
        mean_rsq_train,
        mean_acc_train,
        mean_loss_val,
        mean_rsq_val,
        mean_acc_val
):
    """
    Logs the epoch statistics
    """
    log_parameters['x'].append(epoch_index)
    log_parameters['loss_train'].append(mean_loss_train)
    log_parameters['r_sq_train'].append(mean_rsq_train)
    log_parameters['acc_train'].append(mean_acc_train)
    log_parameters['loss_val'].append(mean_loss_val)
    log_parameters['r_sq_val'].append(mean_rsq_val)
    log_parameters['acc_val'].append(mean_acc_val)
    return log_parameters
