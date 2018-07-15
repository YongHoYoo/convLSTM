import gc
import time
import torch
import pickle 
import argparse
import torchvision
import torch.nn as nn 
from torch import optim
from pathlib import Path 
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend 
import numpy as np
from model import ConvEncDec

class movingMNIST(Dataset):
    def __init__(self, data):
        self.input = data[:10]
        self.output = data[10:20]

    def __getitem__(self, idx):
        return self.input[:,idx].to('cuda'), self.output[:,idx].to('cuda') 

    def __len__(self):
        return self.input.size(1) 
    

if __name__=='__main__': 
  
    parser = argparse.ArgumentParser(description='Argument Parser') 

    parser.add_argument('--bsz', type=int, default=5) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--ksz', type=int, default=5) 
    parser.add_argument('--nhid', type=list, default=[128,64,64]) 
    parser.add_argument('--dropout', type=float, default=0.2) 
    parser.add_argument('--h_dropout', type=float, default=0.0) 
    parser.add_argument('--save_folder', type=str, default='image') 
    parser.add_argument('--gate', action='store_true') 

    args = parser.parse_args() 

    # make save_folder 
    save_folder = Path(args.save_folder) 
    save_folder.mkdir(parents=True, exist_ok=True)    
   
    dataset = np.load('../../mnist_test_seq.npy')
    dataset = torch.tensor(dataset, dtype=torch.float32) # 20 by 10000 by 64 by 64
    dataset/=255 # normalize
    
    trainset = dataset[:,:100,:,:].unsqueeze(2)  # 20 by 7000 by 1 by 64 by 64
    validset  = dataset[:,100:200,:,:].unsqueeze(2)  # 20 by 3000 by 1 by 64 by 64
 
    model = ConvEncDec(1, args.nhid, args.ksz, dropout=args.dropout, h_dropout=args.h_dropout, gate=args.gate).to('cuda')    

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6) 

    trainset = movingMNIST(trainset) 
    validset = movingMNIST(validset) 

    train_loader = DataLoader(dataset=trainset, batch_size=5, shuffle=True) 
    valid_loader = DataLoader(dataset=validset, batch_size=5, shuffle=False) 

    all_valid_loss = [] 

    for epoch in range(args.epochs): 

        all_time = time.time() 

        model.train() 
        print('*** epoch %d ***'%epoch) 
        print('----------------') 

        for i, data in enumerate(train_loader):
            start_time = time.time() 
    
            optimizer.zero_grad() 

            input, target = data

            targets = torch.cat([input, target], 1) 

            outputs, hidden = model(input, target) 
            loss = criterion(outputs, targets) 

            loss.backward() 
            optimizer.step() 

        optimizer.param_groups[0]['lr']*=0.9 

        model.eval() 
        valid_loss = [] 

        for i, data in enumerate(valid_loader):

            input, target = data 
            targets = torch.cat([input, target], 1) 
            outputs, hidden = model(input, target) 

            loss = criterion(outputs, targets) 
            valid_loss.append(loss.item()) 
            if i%2==1: 
                print('%s %d/%d:%7.5f'%(args.save_folder, i,100/5,sum(valid_loss)/len(valid_loss)))

            if i==0: # 5 6 1 64 64
                disp_outputs = outputs[0,:,0].detach().to('cpu') # 6 64 64 
                disp_outputs = torch.stack(disp_outputs.chunk(disp_outputs.size(0),0),2).view(64,-1)

                disp_targets = targets[0,:,0].detach().to('cpu') 
                disp_targets = torch.stack(disp_targets.chunk(disp_targets.size(0),0),2).view(64,-1) 
                save_filename = str(save_folder.joinpath('image%d.png'%epoch)) 
                torchvision.utils.save_image(torch.cat([disp_targets, disp_outputs], 0), save_filename) 

        all_valid_loss.append(sum(valid_loss)/len(valid_loss))
        pickle.dump(all_valid_loss, open(str(save_folder.joinpath('valid.pkl')), 'wb'))
