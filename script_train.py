import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import eval_utils
import train_utils
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from dataLoaders.PitchContourDataset import PitchContourDataset
from models.PCAssessNet import PitchRnn, PitchCRnn

# Training settings
parser = argparse.ArgumentParser(description='Assign3: Performance Asssessment')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.01,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, metavar='M',
#                     help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Weight decay hyperparameter')
parser.add_argument('--num_epochs', type=int, metavar='N', default=50,
                    help='number of epochs to train')
parser.add_argument('--model', default='pitchrnn',
                    choices=['pitchrnn', 'pitchrnn'],
                    help='which model to train / evaluate')
parser.add_argument('--metric', type=int, default=1,
                    choices=[0, 1, 2, 3],
                    help='0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality')
parser.add_argument('--save-dir', default='models/')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set manual random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# check if cuda is available and print result
CUDA_AVAILABLE = args.cuda
print('Running on GPU: ', CUDA_AVAILABLE)

# initialize dataset, dataloader and created batched data
BAND = 'combined'
SEGMENT = '2'
NUM_DATA_POINTS = 1000
NUM_BATCHES = 10
METRIC = args.metric
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)

dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
training_data, _, validation_data, _, _, _ = dataloader.create_split_data(chunk_len=2000, hop=500)

# Augment data
training_data = train_utils.augment_data(training_data)

# initialize the model
perf_model = PitchRnn()
if args.model == 'pitchcrnn':
    perf_model = PitchCRnn()
if CUDA_AVAILABLE:
    perf_model.cuda()
print(perf_model)

# define loss criterion
criterion = nn.MSELoss()

# initialize training hyperparamaters 
# you may want to tune these hyperparameters from the argparser during training
NUM_EPOCHS = args.num_epochs
LR_RATE = args.lr
W_DECAY = args.weight_decay
perf_optimizer = optim.Adam(perf_model.parameters(), lr=LR_RATE, weight_decay=W_DECAY)
# declare save file name
file_info = args.model + '_' + str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC)

## define training parameters
PRINT_EVERY = 1
ADJUST_EVERY = 15
START = time.time()
best_val_loss = 1.0

# train and validate
try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    log_parameters = train_utils.log_init()
    for epoch in range(1, NUM_EPOCHS + 1):
        # perform training and validation
        train_loss, train_r_sq, train_accu, val_loss, val_r_sq, val_accu = train_utils.train_and_validate(perf_model, criterion, perf_optimizer,
                                                                                                          training_data, validation_data, METRIC)

        # adjut learning rate
        train_utils.adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)

        # log data for visualization later
        log_parameters = train_utils.log_epoch_stats(
            log_parameters,
            epoch,
            train_loss,
            train_r_sq,
            train_accu,
            val_loss,
            val_r_sq,
            val_accu
        )

        # print loss
        if epoch % PRINT_EVERY == 0:
            print('[%s (%d %.1f%%)]' % (train_utils.time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            print('[%s %0.5f, %s %0.5f, %s %0.5f]' % ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu))
            print('[%s %0.5f, %s %0.5f, %s %0.5f]' % ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu))

        # save model if best validation loss
        if val_loss < best_val_loss:
            n = file_info + '_best'
            train_utils.save(n, perf_model)
            best_val_loss = val_loss

    print("Saving...")
    train_utils.save(file_info, perf_model, log_parameters)

except KeyboardInterrupt:
    print("Saving before quit...")
    train_utils.save(file_info, perf_model, log_parameters)

# RUN VALIDATION SET ON THE BEST MODEL
# read the best model
filename = file_info + '_best' + '_Reg'
if torch.cuda.is_available():
    perf_model.cuda()
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt', map_location=lambda storage, loc: storage))

# run on validation set
val_loss, val_r_sq, val_accu = eval_utils.eval_model(perf_model, criterion, validation_data, METRIC)
print('[%s %0.5f, %s %0.5f, %s %0.5f]' % ('Best Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu))
