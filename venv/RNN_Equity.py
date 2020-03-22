"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import LoadTrade_normalize as LT
# torch.manual_seed(1)    # reproducible

ReviewDays = 120
# Hyper Parameters
TIME_STEP = 50  # rnn time step
INPUT_SIZE = 5 * ReviewDays  # rnn input size
LR = 0.02  # learning rate

#load Data
lt = LT.loader()
lt.load()
lt.genTrainData()
trainData = torch.from_numpy(lt.trainData)
target = torch.from_numpy(lt.target)

# show data

x_np = np.reshape(trainData, (-1, INPUT_SIZE))
y_np = target
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    state = None
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=INPUT_SIZE * 20,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(INPUT_SIZE * 20, 3)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        self.state = h_state
        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs

def save():
    rnn = RNN()
    print(rnn)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    h_state = None  # for initial hidden state

    plt.figure(1, figsize=(12, 5))
    plt.ion()  # continuously plot

    for step in range(100):


        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        prediction, h_state = rnn(x, h_state)  # rnn output
        # !! next step is important !!
        h_state = h_state.data  # repack the hidden state, break the connection from last iteration

        loss = loss_func(prediction, y)  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        plotting
        plt.plot(steps, y_np[:,2].flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy()[:,2].flatten(), 'b-')
        plt.draw();
        plt.pause(0.1)

    plt.ioff()
    plt.show()

    torch.save(rnn, 'reg_net.pkl')  # save entire net

#Restore
def restore_net():
    # restore entire net1 to net2
    rnn2 = torch.load('reg_net.pkl')
    h_state = rnn2.state
    start, end = 0, 1 * np.pi  # time range
    steps = np.linspace(start, end, 10, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor

    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)

    prediction, h_state = rnn2(x, h_state)
    # plotting
    plt.plot(steps, y_np[:,2].flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy()[:,2].flatten(), 'b-')
    plt.draw();
    plt.pause(5)


plt.ioff()
plt.show()

#main
save()

# restore_net()

