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

class PCB(Dataset):
    def __init__(self, data, mask):  # data: 9 by 20 by 1 by 64 by 64 
        self.input = data[:, :10] 
        self.output = data[:, 10:20] 

        self.imask = mask[:, :10] 
        self.omask = mask[:, 10:20] 

    def __getitem__(self, idx): 
        return self.input[idx].to('cuda'), self.output[idx].to('cuda'), self.imask[idx].to('cuda'), self.imask[idx].to('cuda') 

    def __len__(self):
        return self.input.size(0) 
    
if __name__=='__main__': 
  
    parser = argparse.ArgumentParser(description='Argument Parser') 

    parser.add_argument('--bsz', type=int, default=5) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--ksz', type=int, default=5) 
    parser.add_argument('--nhid', type=list, default=[128,64,64]) 
    parser.add_argument('--dropout', type=float, default=0.1) 
    parser.add_argument('--h_dropout', type=float, default=0.1) 
    parser.add_argument('--train_folder', type=str, default='train') 
    parser.add_argument('--valid_folder', type=str, default='valid') 
    parser.add_argument('--gate', action='store_true') 

    args = parser.parse_args() 

    # make save_folder 
    train_folder = Path(args.train_folder) 
    train_folder.mkdir(parents=True, exist_ok=True)    
 
    valid_folder = Path(args.valid_folder) 
    valid_folder.mkdir(parents=True, exist_ok=True)    

    image = pickle.load(open('../../pcb_norm_image.pkl', 'rb')) 
    image = torch.stack(image.chunk(10,0),0) 

    bmask = pickle.load(open('../../pcb_mask.pkl', 'rb')) 
    bmask = torch.stack(bmask.chunk(10,0),0) 
    
    dataset = []
    bmasks = [] 

    for i in range(len(image)-1): 
        dataset.append(torch.cat([image[i], image[i+1]], 0)) #20 1 64 64
        bmasks.append(torch.cat([bmask[i], bmask[i+1]], 0)) 

    dataset = torch.stack(dataset, 0)
    bmasks = torch.stack(bmasks, 0).unsqueeze(2) 


    trainset = dataset[:5] 
    trainmask = bmasks[:5] 

    validset = dataset[5:] 
    validmask = bmasks[5:] 
    
    trainset = PCB(trainset, trainmask)
    validset = PCB(validset, validmask) 

    train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False) 
    valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=False) 
 
    model = ConvEncDec(1, args.nhid, args.ksz, dropout=args.dropout, h_dropout=args.h_dropout, gate=args.gate).to('cuda')    

    criterion = nn.MSELoss() 
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-6) 

    all_valid_loss = [] 

    for epoch in range(args.epochs): 

        model.train() 
        hidden = None 
        print('*** epoch %d ***'%epoch) 
        print('----------------') 
        
        for i, data in enumerate(train_loader): 

            optimizer.zero_grad() 

            input, target, imask, tmask = data 
                       
            targets = torch.cat([input, target], 1) 
            masks = torch.cat([imask, tmask], 1)    

            outputs, hidden = model(input, target, hidden) 

            m_outputs = outputs*masks
            m_targets = targets*masks
            loss = criterion(m_outputs, m_targets) 
            loss.backward() 
            optimizer.step() 

            new_hidden = [] 
            for j in range(len(hidden)):
                h,c = hidden[j][0], hidden[j][1] 
                h = h.detach()
                c = c.detach() 
                new_hidden.append((h,c)) 

            hidden = new_hidden 

            if i==0: # 5 6 1 64 64
                disp_outputs = m_outputs[0,:,0].detach().to('cpu') # 6 64 64 
                disp_outputs = torch.stack(disp_outputs.chunk(disp_outputs.size(0),0),2).view(64,-1)

                disp_targets = m_targets[0,:,0].detach().to('cpu') 
                disp_targets = torch.stack(disp_targets.chunk(disp_targets.size(0),0),2).view(64,-1) 
                save_filename = str(train_folder.joinpath('image%d.png'%epoch)) 
                torchvision.utils.save_image(torch.cat([disp_targets, disp_outputs], 0), save_filename) 


        optimizer.param_groups[0]['lr']*=0.9

        model.eval() 
        
        valid_loss = [] 
        for i, data in enumerate(valid_loader): 
            input, target, imask, tmask = data 
            
            targets = torch.cat([input, target], 1) 
            masks = torch.cat([imask, tmask], 1) 
            outputs, hidden = model(input, target, hidden) 

            m_outputs = outputs*masks
            m_targets = targets*masks
            
            loss = criterion(m_outputs, m_targets) 
            valid_loss.append(loss.item()) 

            if i==3: 
                print('%s %d/%d:%7.5f'%(args.valid_folder, i,4,sum(valid_loss)/len(valid_loss)))
            
            if i==0: # 5 6 1 64 64
                disp_outputs = m_outputs[0,:,0].detach().to('cpu') # 6 64 64 
                disp_outputs = torch.stack(disp_outputs.chunk(disp_outputs.size(0),0),2).view(64,-1)

                disp_targets = m_targets[0,:,0].detach().to('cpu') 
                disp_targets = torch.stack(disp_targets.chunk(disp_targets.size(0),0),2).view(64,-1) 
                save_filename = str(valid_folder.joinpath('image%d.png'%epoch)) 
                torchvision.utils.save_image(torch.cat([disp_targets, disp_outputs], 0), save_filename) 


