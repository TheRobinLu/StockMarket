import LoadTrade_normalize as LT
import torch
from torch import nn

import numpy
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10000               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 120
TIME_STEP = 30          # rnn time step / image height
INPUT_SIZE = 5         # rnn input size / image width
OUTPUT_SIZE = 3
HIDEN_SIZE = 50
LR = 0.001               # learning rate
DROPOUT = 0.2


#   Variable
# realout = []
#
# realtarget = []
#   Load Data

lt = LT.loader()
lt.period = TIME_STEP
lt.load()
lt.genTrainData()
lt.normalize()
trainData = torch.from_numpy(lt.trainData)
target = torch.from_numpy(lt.target)

closeValue = numpy.array(lt.closeData)


print(trainData.size())
print(target.size())

#   Build RNN
class RNN(nn.Module):
    h_n = None
    h_c = None
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=HIDEN_SIZE,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            dropout=0.3,
            batch_first=False,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(HIDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (self.h_n, self.h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()
print(rnn)
optimizer = torch.optim.Adam (rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()                       # the target label is not one-hotted

#   Save
def Save():
    for epoch in range(EPOCH):
        # for i in range(len(trainData)):  # gives batch data
        x = trainData#[i, :]
        y = target#[i,:]
        v_x = x.view(-1,TIME_STEP,INPUT_SIZE).float()
        v_y = y.view(-1,OUTPUT_SIZE).float()
        optimizer.zero_grad()  # clear gradients for this training step
        output = rnn(v_x)
        loss = loss_func(output, v_y)  # cross entropy loss

        # loss = loss_func()
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        realout = []
        realtarget = []

        if (epoch) % 20 == 0 :
            k = numpy.random.randint (len(output), size= 1)
            k = k[0]
            for i in range(len(output[k])):
                realout.append( output[k,i] * closeValue[k])

            for i in range(3):
                realtarget.append(numpy.float(v_y[k,  i]  * closeValue[k]))
            print(epoch, k, numpy.array(realout, dtype=float), realtarget, numpy.float(loss))

            # realout = []
            # realtarget = []

    torch.save(net1, 'Stock.pkl')  # save entire net

#   Restore
def Restore():
    net = torch.load('Stock.pkl')


Save()






