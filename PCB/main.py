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

class ConvLSTMCell(nn.Module): 
    def __init__(self, cinp, chid, ksz): 
        
        super(ConvLSTMCell, self).__init__() 
        
        self.cinp = cinp
        self.chid = chid 
        self.ksz = ksz
        pad = int((ksz-1)/2) 

        self.i2h = nn.Conv2d(cinp, 4*chid, ksz, 1, pad, bias=True) 
        self.h2h = nn.Conv2d(chid, 4*chid, ksz, 1, pad) 

    def forward(self, input, hidden=None): 

        bsz, _, height, width = input.size() 

        if hidden is None: 
            hidden = self.init_hidden(bsz, height, width) 

        hx, cx = hidden

        iGates = self.i2h(input)
        hGates = self.h2h(hx) 

        state = fusedBackend.LSTMFused.apply 
        hy, cy = state(iGates, hGates, cx) 

        return hy,cy 

    def init_hidden(self, bsz, height, width): 
        weight = next(self.parameters()).data 

        return (weight.new(bsz, self.chid, height, width).zero_(), #.requires_grad_(), 
                weight.new(bsz, self.chid, height, width).zero_())#.requires_grad_()) 

class ConvEncoder(nn.Module): 
    def __init__(self, cinp, chids, ksz, dropout=0.2, h_dropout=0.0):
        super(ConvEncoder, self).__init__() 

        self.cinps = [cinp] + chids # 1 128 64 64
        self.chids = chids # 128 64 64 
        self.nlayers = len(chids) # 3
        self.ksz = ksz 
        self.dropout = dropout
        self.h_dropout = h_dropout 

        pad = int((ksz-1)/2) 
        
        self.layer_stack = nn.ModuleList([
            ConvLSTMCell(self.cinps[i], self.chids[i], ksz) 
            for i in range(self.nlayers)]) 

        self.top = nn.Conv2d(chids[-1], cinp, ksz, 1, pad, bias=True) 

    def forward(self, input, hidden=None):

        bsz, steps, cinp, height, width = input.size() 

        if hidden==None: 
            hidden = [layer.init_hidden(bsz, height, width) 
                for layer in self.layer_stack] 
    
        if self.dropout>0 and self.training: 
            mask = [input.new(bsz, cinp, height, width).bernoulli_(1-self.dropout).div(1-self.dropout)]
            for i in range(len(hidden)):
                mask.append(hidden[i][0].new(hidden[i][0].size()).bernoulli_(1-self.dropout).div(1-self.dropout)) 
            self.mask = mask 
        else: 
            self.mask = None 

        if self.h_dropout>0 and self.training:
            mask = []
            for i in range(len(hidden)):
                mask.append(hidden[i][0].new(hidden[i][0].size()).bernoulli_(1-self.dropout).div(1-self.dropout)) 
            self.h_mask = mask 
        else:
            self.h_mask = None 

        for step in range(steps): 
            next_hidden = [] 
            
            # mask all of input. 
            x = input[:,step] 

            if self.mask is not None: # training & dropout 
                x = x*self.mask[0] 
            
            for i, layer in enumerate(self.layer_stack): 
                
                h,c = hidden[i]
                
                if self.h_mask is not None: 
                    h = h*self.h_mask[i]         

                x,c = layer(x, (h,c)) 
                next_hidden.append((x,c)) 
                
                if self.mask is not None: 
                    x = x*self.mask[i+1]
            
            hidden = next_hidden

        return hidden 


class ConvDecoder(nn.Module):
    def __init__(self, cinp, chids, ksz, reverse=False, dropout=0.2, h_dropout=0.0): 
     
        super(ConvDecoder, self).__init__() 
   
        self.cinps = [cinp] + chids 
        self.chids = chids 
        self.nlayers = len(chids) 
        self.ksz = ksz 
        self.reverse = reverse 
        self.dropout = dropout
        self.h_dropout = h_dropout

        pad = int((ksz-1)/2) 

        self.layer_stack = nn.ModuleList([
            ConvLSTMCell(self.cinps[i], self.chids[i], ksz) 
            for i in range(self.nlayers)]) 

        self.top = nn.Conv2d(chids[-1], cinp, ksz, 1, pad, bias=True) 

    def forward(self, hidden, target): 

        if target is None: # free running 
            pass
        else: # teacher forcing 
            pass

        bsz, steps, cinp, height, width = target.size()

        if self.reverse:
            timesteps = range(steps-1,0,-1) 
        else:
            timesteps = range(steps-1) 

        if self.dropout>0 and self.training: 
            mask = [input.new(bsz, cinp, height, width).bernoulli_(1-self.dropout).div(1-self.dropout)]
            for i in range(len(hidden)):
                mask.append(hidden[i][0].new(hidden[i][0].size()).bernoulli_(1-self.dropout).div(1-self.dropout)) 
            self.mask = mask 
        
        else: 
            self.mask = None 

        if self.h_dropout>0 and self.training:
            mask = []
            for i in range(len(hidden)):
                mask.append(hidden[i][0].new(hidden[i][0].size()).bernoulli_(1-self.dropout).div(1-self.dropout)) 
            self.h_mask = mask 
        else:
            self.h_mask = None 


        outputs = [self.top(hidden[-1][0])]

        for step in timesteps: 
            next_hidden = [] 
            x = target[:, step]

            # mask of target
            if self.mask is not None: 
                x = x*self.mask[0] 
            
            for i, layer in enumerate(self.layer_stack): 
                
                h, c = hidden[i] 

                if self.h_mask is not None: 
                    h =h*self.h_mask[i] 
        
                x,c = layer(x, (h,c)) 
                next_hidden.append((x,c)) 
                
                if self.mask is not None: 
                    x = x*self.mask[i+1] 
            
            hidden = next_hidden 
            outputs.append(self.top(x)) 

        if self.reverse:
            outputs = torch.stack(outputs[::-1], 1) 
        else:
            outputs = torch.stack(outputs, 1) 
           
        return outputs
    

class ConvEncDec(nn.Module): 
    def __init__(self, cinp, chids, ksz, dropout=0.2, h_dropout=0.0): 
        super(ConvEncDec, self).__init__() 

        self.dropout = dropout 
        self.h_dropout = h_dropout

        self.convEncoder = ConvEncoder(cinp, chids, ksz, dropout=dropout, h_dropout=h_dropout) 
        self.convReconstructor= ConvDecoder(cinp, chids, ksz, reverse=True, dropout=dropout, h_dropout=h_dropout) 
        self.convPredictor = ConvDecoder(cinp, chids, ksz, dropout=dropout, h_dropout=h_dropout) 

    def forward(self, input, target, hidden=None):

        # make dropout mask. !

        hidden = self.convEncoder(input, hidden)
        reconstructed = self.convReconstructor(hidden, input) 
        predicted = self.convPredictor(hidden, target) 
        output = torch.cat([reconstructed, predicted], 1) 

        return output, hidden

    
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

    args = parser.parse_args() 

    # make save_folder 
    train_folder = Path(args.train_folder) 
    train_folder.mkdir(parents=True, exist_ok=True)    
 
    valid_folder = Path(args.valid_folder) 
    valid_folder.mkdir(parents=True, exist_ok=True)    

    image = pickle.load(open('../../pcb_image.pkl', 'rb')) 
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
 
    model = ConvEncDec(1, args.nhid, args.ksz, dropout=args.dropout, h_dropout=args.h_dropout).to('cuda')    

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) 

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
            
#            outputs = masks*outputs + (1-masks)*torch.cat([input,target],1)  

            loss = criterion(outputs, targets)  
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
                disp_outputs = outputs[0,:,0].detach().to('cpu') # 6 64 64 
                disp_outputs = torch.stack(disp_outputs.chunk(disp_outputs.size(0),0),2).view(64,-1)

                disp_targets = targets[0,:,0].detach().to('cpu') 
                disp_targets = torch.stack(disp_targets.chunk(disp_targets.size(0),0),2).view(64,-1) 
                save_filename = str(train_folder.joinpath('image%d.png'%epoch)) 
                torchvision.utils.save_image(torch.cat([disp_targets, disp_outputs], 0), save_filename) 


        if epoch<50: 
            optimizer.param_groups[0]['lr']*=0.9

        model.eval() 
        
        valid_loss = [] 
        for i, data in enumerate(valid_loader): 
            input, target, imask, tmask = data 
            
            targets = torch.cat([input, target], 1) 
            masks = torch.cat([imask, tmask], 1) 
            outputs, hidden = model(input, target, hidden) 

 #           outputs = masks*outputs + (1-masks)*torch.cat([input,target],1) 
          
            loss = criterion(outputs, targets) 
            valid_loss.append(loss.item()) 

            print('%d/%d:%7.5f'%(i,100/5,sum(valid_loss)/len(valid_loss)))
            
            if i==0: # 5 6 1 64 64
                disp_outputs = outputs[0,:,0].detach().to('cpu') # 6 64 64 
                disp_outputs = torch.stack(disp_outputs.chunk(disp_outputs.size(0),0),2).view(64,-1)

                disp_targets = targets[0,:,0].detach().to('cpu') 
                disp_targets = torch.stack(disp_targets.chunk(disp_targets.size(0),0),2).view(64,-1) 
                save_filename = str(valid_folder.joinpath('image%d.png'%epoch)) 
                torchvision.utils.save_image(torch.cat([disp_targets, disp_outputs], 0), save_filename) 


