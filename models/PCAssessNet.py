import torch.nn as nn


class PitchRnn(nn.Module):
    """
    Class to implement a deep RNN model for music performance assessment using
    pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchRnn class with internal parameters for the different layers
        This should be a RNN based model. You are free to choose your hyperparameters with 
        respect type of RNN cell (LSTM, GRU), type of regularization (dropout, batchnorm)
        and type of non-linearity
        """
        super(PitchRnn, self).__init__()
        self.hidden_size = 16
        self.n_layers = 1
        self.features = 16
        self.fc = nn.Sequential(nn.Linear(1, self.features),
                                # nn.ReLU()     # model performs better without activation function
                                )
        self.gru = nn.GRU(input_size=self.features, hidden_size=self.hidden_size, num_layers=self.n_layers, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        """
        Defines the forward pass of the module
        Args:
            input: 	torch Variable (mini_batch_size x seq_len), of input pitch contours
        Returns:
            output: torch Variable (mini_batch_size), containing predicted ratings
        """
        self.hidden = self.init_hidden(input.size()[0])
        output = self.fc(input.unsqueeze(2))
        output, self.hidden = self.gru(output, self.hidden)
        output = self.fc2(output)
        return output[:, -1, -1]

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        self.hidden = None


class PitchCRnn(nn.Module):
    """
    Class to implement a deep CRNN model for music performance assessment using
    pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchCRnn class with internal parameters for the different layers
        This should be a RNN based model. You are free to choose your hyperparameters with 
        respect type of RNN cell (LSTM, GRU), type of regularization (dropout, batchnorm)
        and type of non-linearity. Since this is a CRNN model, you will also need to decide 
        how many convolutonal layers you want to take as input. Note that the minimum
        sequence length that you should expect is 2000.
        """
        super(PitchCRnn, self).__init__()
        self.hidden_size = 64
        self.layer1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=16, stride=10, dilation=1),
                                    nn.BatchNorm1d(16), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=8, stride=4), nn.Dropout(0.25))
        self.layer2 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=8, stride=4, dilation=1),
                                    nn.BatchNorm1d(32), nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2), nn.Dropout(0.25))
        self.gru = nn.GRU(input_size=32, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(self.hidden_size, 16), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(16, 1)

    def forward(self, input):
        """
        Defines the forward pass of the module
        Args:
            input: 	torch Variable (mini_batch_size x seq_len), of input pitch contours
        Returns:
            output: torch Variable (mini_batch_size), containing predicted ratings
        """
        output = self.layer1(input.unsqueeze(1))
        output = self.layer2(output)
        output = output.transpose(1, 2)
        output, self.hidden = self.gru(output, self.hidden)
        output = self.fc(output)
        output = self.fc2(output)
        return output[:, -1, -1]

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the recurrent layers
        Args:
            mini_batch_size: number of data samples in the mini-batch
        """
        self.hidden = None
