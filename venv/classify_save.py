
"""

"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible
plt.figure(1, figsize=(12, 6))


# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    #Data
    seeds = 200
    scattr = 2
    n_data = torch.ones(seeds, 2)
    x0 = torch.normal(scattr * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    # print(n_data)
    # print(x0)

    y0 = torch.zeros(seeds)  # class0 y data (tensor), shape=(100, 1)
    # print(y0)
    x1 = torch.normal(-scattr * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(seeds)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2),
    )
    optimizer = torch.optim.Adam (net1.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

    plt.ion()  # something about plotting

    for t in range(300):
        out = net1(x)  # input x and predict based on x
        loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients


    plt.subplot(231)
    plt.title('Net1')

    # plt.cla()
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdBu_r')
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color': 'blue'})

    # plt.ioff()
    # plt.show()
    plt.subplot(234)
    plt.title('Net1_real')
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='PRGn_r')

    # 2 ways to save the net
    torch.save(net1, 'net.pkl')  # save entire net
    torch.save(net1.state_dict(), 'net_params.pkl')   # save only the parameters


def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    seeds = 100
    scattr = 5
    n_data = torch.ones(seeds, 2)
    x0 = torch.normal(scattr * n_data, 2)  # class0 x data (tensor), shape=(100, 2)
    # print(n_data)
    print(x0)

    y0 = torch.zeros(seeds)  # class0 y data (tensor), shape=(100, 1)
    # print(y0)
    x1 = torch.normal(-scattr * n_data, 2)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(seeds)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

    out = net2(x)
    # plt.cla()

    plt.subplot(232)
    plt.title('Net2')


    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdBu_r')
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color': 'blue'})
    plt.subplot(235)
    plt.title('Net2_real')
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=target_y, s=10, lw=0, cmap='PRGn_r')


def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    seeds = 20
    scattr = 2
    n_data = torch.ones(seeds, 2)

    x0 = torch.normal(scattr-1 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(seeds)  # class0 y data (tensor), shape=(100, 1)
    # print(y0)
    x1 = torch.normal(-scattr * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(seeds)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

    out = net3(x)

    # plot result
    plt.subplot(233)
    plt.title('Net3')
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdBu_r')
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 10, 'color': 'blue'})



    plt.subplot(236)
    plt.title('Net3_real')
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=target_y, s=10, lw=0, cmap='PRGn_r')

    plt.ioff()
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()