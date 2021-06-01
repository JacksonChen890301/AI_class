import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from pylab import meshgrid
from sklearn.metrics import mean_squared_error

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
np.random.seed(7)
SCALER_X = MinMaxScaler()
SCALER_R = StandardScaler()

class Projectile(Dataset):
    def __init__(self, size=500, train=True):
        self.train = train
        self.train_size = int(size*0.8)
        self.test_size = int(size*0.2)
        v0 = np.random.uniform(1, 100, size=(size, 1))
        theta = np.random.uniform(0, np.pi/2, size=(size, 1))
        # theta = np.linspace(0, np.pi/2, 50).reshape(50, 1)
        # for i in range(39):
        #     t = np.linspace(0, np.pi/2, 50).reshape(50, 1)
        #     theta = np.append(theta, t, axis=0)
        r = SCALER_R.fit_transform(projectile_formula(v0, theta, mode='range'))
        # h = SCALER_R.fit_transform(projectile_formula(v0, theta, mode='height'))
        self.x_train, self.y_train = SCALER_X.fit_transform(np.concatenate((v0[:int(size*0.8), :], theta[:int(size*0.8), :]), axis=1)), r[:int(size*0.8), :]
        self.x_val, self.y_val = SCALER_X.transform(np.concatenate((v0[int(size*0.8):, :], theta[int(size*0.8):, :]), axis=1)), r[int(size*0.8):, :]

    def __getitem__(self, index):
        if self.train:
            inputs, target = self.x_train[index, :], self.y_train.reshape(self.train_size, 1)[index]
        else:
            inputs, target = self.x_val[index, :], self.y_val.reshape(self.test_size, 1)[index]
        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        target = torch.from_numpy(np.array(target)).type(torch.FloatTensor)
        return inputs, target

    def __len__(self):
        if self.train:
            return len(self.x_train)
        else:
            return len(self.x_val)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        unit = 8
        self.linear1 = nn.Linear(2, unit)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.normal_(self.linear1.bias, std=1e-6)
        # self.linear2 = nn.Linear(unit, unit)
        # torch.nn.init.xavier_uniform_(self.linear2.weight)
        # torch.nn.init.normal_(self.linear2.bias, std=1e-6)
        self.linear3 = nn.Linear(unit, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def main():
    # train_data = Projectile(train=True)
    # test_data = Projectile(train=False)
    # train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0)
    # test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=0)

    # epoch = 500
    # tl, vl = [], []
    # model = Net().cpu()
    # loss_func = nn.MSELoss()
    # optim = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=3e-3)
    # for i in range(epoch):
    #     train_loss, val_loss = 0.0, 0.0
    #     t, v = 0, 0
    #     model.train()
    #     for inputs, labels in train_loader:
    #         y_pred = model(inputs)
    #         loss = loss_func(y_pred, labels)
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #         train_loss += loss.item()
    #         t += 1
    #     model.eval()
    #     for inputs, labels in test_loader:
    #         y_pred = model(inputs)
    #         loss = loss_func(y_pred, labels)
    #         val_loss += loss.item()
    #         v += 1
    #     print('epoch: {}, Train Loss: {:.6f}, Eval Loss: {:.6f}'.format(i+1, train_loss/t, val_loss/v))
    #     tl.append(train_loss / t)
    #     vl.append(val_loss / v)

    # plt.plot(np.linspace(1, epoch, epoch), tl, c='green', label='train')
    # plt.plot(np.linspace(1, epoch, epoch), vl, c='red', label='validate')
    # plt.xlabel('epoch')
    # plt.ylabel('loss(MSE)')
    # plt.legend()
    # plt.show()
    # plt.close()

    # fig, ax = plt.subplots()
    # plt.subplot(1, 2, 1)
    # v0 = np.linspace(0, 150, 1000)
    # theta = np.linspace(-np.pi/2, np.pi, 1000)
    # X,Y = meshgrid(v0, theta)
    # rang = projectile_formula(X, Y, mode='range')
    # print(X.shape)
    # plt.imshow(rang, cmap='hot', vmax=1000, vmin=0)
    # plt.colorbar()
    # plt.xlabel('initial velocity')
    # plt.ylabel('angle')
    # plt.xticks([0, 500, 1000], ['0', '75', '150'])
    # plt.yticks([0, 1000/3 ,2000/3, 1000], ['-90', '0', '90','180'])

    # fig, ax = plt.subplots()
    # plt.subplot(1, 2, 1)
    # v0 = np.linspace(0, 150, 1000)
    # theta = np.linspace(-np.pi/2, np.pi, 1000)
    # X,Y = meshgrid(v0, theta)
    # rang = projectile_formula(X, Y, mode='height')
    # print(X.shape)
    # plt.imshow(rang, cmap='hot', vmax=1100, vmin=0)
    # plt.colorbar()
    # plt.xlabel('initial velocity')
    # plt.ylabel('angle')
    # plt.xticks([0, 500, 1000], ['0', '75', '150'])
    # plt.yticks([0, 1000/3 ,2000/3, 1000], ['-90', '0', '90','180'])

    # plt.subplot(1, 2, 2)
    # x = SCALER_X.transform(np.concatenate((X.reshape(1000,1000,1), Y.reshape(1000,1000,1)), axis=2).reshape(1000000, 2))
    # x = torch.from_numpy(x).type(torch.FloatTensor)
    # with torch.no_grad():
    #     y_pred = SCALER_R.inverse_transform(model(x).detach().numpy().reshape(-1, 1)).reshape(1000, 1000)
    #     plt.imshow(y_pred, cmap='hot', vmax=1100, vmin=0)
    #     plt.colorbar()
    #     plt.xlabel('initial velocity')
    #     plt.ylabel('angle')
    #     plt.xticks([0, 500, 1000], ['0', '75', '150'])
    #     plt.yticks([0, 1000/3 ,2000/3, 1000], ['-90', '0', '90','180'])

    # plt.subplot(1, 3, 3)
    # plt.imshow(np.abs(y_pred-rang), cmap='hot')
    # plt.colorbar()
    # plt.xlabel('initial velocity')
    # plt.ylabel('angle')
    # plt.xticks([0, 500, 1000], ['0', '50', '100'])
    # plt.yticks([0, 500, 1000], ['0', '45', '90'])
    # plt.show()
    # plt.close()
    # print(mean_squared_error(rang.reshape(-1, 1), y_pred.reshape(-1, 1)))
    mse = np.array([[206.9, 133.5, 133.3, 135.8], 
                    [102.7, 93.02, 126.9, 127.7],
                    [107.5, 140.5, 91.06, 139.3],
                    [140.2, 228.8, 169.3, 106.4],
                    [146.5, 113.9, 143.2, 139.7]], dtype=float)
    plt.errorbar([16, 32, 64, 128], np.mean(mse, axis=0), np.std(mse, axis=0), fmt='b-o')
    plt.xlabel('batchsize')
    plt.ylabel('loss(MSE)')
    plt.ylim([0, 700])
    plt.show()
    plt.close()
    


def projectile_formula(v0, theta, mode):
    if mode == 'range':
        r = (v0**2)*np.sin(2*theta)/10
        return r
    elif mode == 'height':
        h = ((v0*np.sin(theta))**2)/(2*10)
        return h


if __name__ == '__main__':
    main()


