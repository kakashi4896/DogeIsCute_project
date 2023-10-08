import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nasaomnireader import omnireader
import datetime
import numpy as np
import pandas as pd

class KPData:
    def __init__(self, n):
        self.data = np.zeros((n,))
    def __setitem__(self, key, value):
        self.data[key] = value
    def __getitem__(self, key):
        return self.data[key]
    def __call__(self):
        return self.data

def get_nasa_omni(start_date, end_date, interval):
    intervals = ['5min', '1min','hourly']
    if interval not in intervals:
        raise Exception("interval needs to be 1min, 5min, hourly")
    omniInt = omnireader.omni_interval(start_date,end_date,interval)
    if interval == 'hourly':
        return omniInt
    omniInt_hourly = omnireader.omni_interval(start_date,end_date,'hourly')
    n = len(omniInt['Epoch'])
    KP = KPData(n)
    step = 1
    if interval == '5min':
        step = 12
    elif interval == '1min':
        step = 60
    hourly_KP = omniInt_hourly['KP']
    for i in range(n):
        KP[i] = hourly_KP[i // step]
    omniInt.computed['KP'] = KP
    return omniInt

def get_density(omniInt):
    try:
        return omniInt['N']
    except:
        return omniInt['proton_density']

def get_speed(omniInt):
    try:
        return omniInt['V']
    except:
        return omniInt['flow_speed']

#Create a time window
sTimeIMF = datetime.datetime(2016,1,1)
eTimeIMF = datetime.datetime(2023,8,31)

omniInt = get_nasa_omni(sTimeIMF,eTimeIMF,'1min')

STEP_SIZE = 60

#omniInt['BX_GSE'], omniInt['BY_GSM'], omniInt['BZ_GSM'], omniInt['KP']
# For data preprocess
Bx = omniInt['BX_GSE']
By = omniInt['BY_GSM']
Bz = omniInt['BZ_GSM']
N = get_density(omniInt)
V = get_speed(omniInt)
Kp = omniInt['KP']

df = pd.DataFrame({"Bx": Bx,
                   "By": By,
                   "Bz": Bz,
                   "N": N,
                   "V": V,
                   "Kp": Kp})
# df = df.dropna(how= 'any')
df = df.fillna(method='backfill')
# print(df)
Bx = df['Bx'].values
By = df['By'].values
Bz = df['Bz'].values
N = df['N'].values
V = df['V'].values
Kp = df['Kp'].values[::STEP_SIZE]
B = np.array(list(zip(Bx, By, Bz))).swapaxes(0, 1).reshape(-1, 3, STEP_SIZE)

print ("B shape: ", B.shape)
print ("B: ", B)
#print ("shape: ", Bx.shape, "/", By.shape, "/", Bz.shape, "/", Kp.shape)

import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

def random_seed_setup(seed):
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return 'cuda'
    else:
        return 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

class NASADataset(Dataset):
    def __init__(self,
                 B,
                 mode='train',
                 normalize='none',
                 target_only=False,
                 norm_mean=None,
                 norm_std=None):
        self.mode = mode
        self.data_class = Kp
        data = B.astype(float)
        
        if not target_only:
            feats = [0, 1, 2]
        else:
            feats = [0, 1, 2]

        if mode == 'test':
            # Testing data
            data = data[:, feats, :]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            target = Kp
            data = B
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = list(range(893))
            else:
                indices = list(range(len(data)))
                # only when mode == 'train_all' (i.e., using all data for training)
            
            data = data[indices]
            target = target[indices]
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data)
            self.target = torch.unsqueeze(torch.FloatTensor(target), -1)
        
        self.norm_mean = None
        self.norm_std = None

        if normalize == 'self':
            self.data = np.apply_along_axis(lambda x: ((x - x.mean()) / x.std()), 0,  self.data)

        if normalize == 'given' and (norm_mean is not None) and (norm_std is not None):
            self.data = (self.data - norm_mean) / norm_std
        self.dim = self.data.shape[1]

        print('Finished reading the {} set of the NASA Dataset ({} samples found, each dim = {})'
              .format(mode, self.data.shape[0], self.data.shape[1]))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev', 'train_all']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

def prep_dataloader(B_train, B_test, batch_size, n_jobs=0, target_only=False):
    tr_dataset = NASADataset(
        B_train, mode='train_all',
        normalize='none', 
        target_only=target_only)

    dv_dataset = NASADataset(
        B_train, mode='dev',
        normalize='none', 
        target_only=target_only,
        norm_mean=tr_dataset.norm_mean,
        norm_std=tr_dataset.norm_std)

    tt_dataset = NASADataset(
        B_test, mode='test',
        normalize='none',
        target_only=target_only,
        norm_mean=tr_dataset.norm_mean,
        norm_std=tr_dataset.norm_std)
    
    tr_dataloader = DataLoader(
        tr_dataset, batch_size,
        shuffle=True, drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    
    dv_dataloader = DataLoader(
        dv_dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)

    tt_dataloader = DataLoader(
        tt_dataset, batch_size,
        shuffle=False, drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    
    return tr_dataloader, dv_dataloader, tt_dataloader

class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim, l2_reg=0.):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.RReLU(),  # randomized ReLU
            nn.Linear(16, 32),
            nn.RReLU(),  # randomized ReLU
            nn.Linear(32, 32),
            nn.RReLU(),  # randomized ReLU
            nn.Linear(32, 16),
            nn.RReLU(),  # randomized ReLU
            nn.Linear(16, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')
        self.l2_reg = l2_reg

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)


def cal_loss(pred, target, is_train=True):
    ''' Calculate loss '''
    total_loss = 0

    mse_loss = criterion(pred, target)
    total_loss += mse_loss
    
    return total_loss, mse_loss

def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    loss_record = {'train': []}
    min_loss = 100
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        for i, (x, y) in tqdm(enumerate(tr_set)):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            #print(y.shape, pred.shape)
            total_loss, mse_loss = cal_loss(pred, y, is_train=True)
            total_loss.backward()
            optimizer.step()
            loss_record['train'].append(total_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if min_loss > dev_mse:
            print(f'Saving model (epoch = {epoch}, dev_mae: {dev_mse})')
            torch.save(model.state_dict(), config['save_path'])
            min_loss = dev_mse
        else:
            print(f'(epoch = {epoch}, dev_mae: {dev_mse})')

    print('Finished training after {} epochs'.format(epoch))

    return loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in tqdm(dv_set):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            _, mse_loss = cal_loss(pred, y, is_train=False)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)

    return total_loss

def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds

device = random_seed_setup(42097)
criterion = nn.L1Loss().to(device)#nn.MSELoss(reduction='mean').to(device)
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = True

config = {
    'n_epochs': 2000,                  # maximum number of epochs
    'batch_size': 512,               # mini-batch size for dataloader
    'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 1e-4,                 # learning rate of SGD
        #'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 20,                # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}


tr_set, dv_set, tt_set = prep_dataloader(
    B, B, config['batch_size'], 
    target_only=target_only)

# model = NeuralNet(tr_set.dataset.dim, 0.001).to(device)  # Construct model and move to device
#from model import *
from tsai.all import PatchTST, TSSequencerPlus
# model = PatchTST(B.shape[1], 1, B.shape[2], 1, patch_len=6, attn_dropout=0.3,).to(device)  # Construct model and move to device
model = TSSequencerPlus(B.shape[1], 1, B.shape[2], lstm_dropout=0.3,).to(device)  # Construct model and move to device
model_loss_record = train(tr_set, dv_set, model, config, device)
plot_learning_curve(model_loss_record, title='deep model')
