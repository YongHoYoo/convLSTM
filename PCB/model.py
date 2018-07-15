import torch
import torch.nn as nn 
import torch.nn.functional as f
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend 

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
    def __init__(self, cinp, chids, ksz, dropout=0.2, h_dropout=0.0, gate=False):
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

        if gate is True:
            self.attn = nn.ModuleList([ 
                nn.Conv2d(self.chids[i], 1, ksz, 1, pad, bias=True) 
                for i in range(self.nlayers)])
        else: 
            self.attn = None 

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

        attn_masks = [] 
        for step in range(steps): 
            next_hidden = [] 
            
            # mask all of input. 
            x = input[:,step] 

            if self.mask is not None: # training & dropout 
                x = x*self.mask[0] 
            
            attn_mask = [] 
            for i, layer in enumerate(self.layer_stack): 
                
                h,c = hidden[i]
                
                if self.h_mask is not None: 
                    h = h*self.h_mask[i]         

                x,c = layer(x, (h,c)) 
                
                if self.attn is not None and step<steps-1: # don't apply the attn to last hidden state 
                    attn = self.attn[i](x) 
                    
                    attn_mask.append(attn) 
                    a = nn.Sigmoid()(attn) 
                    x = x*a.expand_as(x) 

                next_hidden.append((x,c)) 
                
                if self.mask is not None: 
                    x = x*self.mask[i+1]
            
            attn_masks.append(attn_mask) 
            hidden = next_hidden

        if self.attn is None: 
            attn_masks = None

        return hidden, attn_masks


class ConvDecoder(nn.Module):
    def __init__(self, cinp, chids, ksz, reverse=False, dropout=0.2, h_dropout=0.0, gate=False): 
     
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

        if gate is True:
            self.attn = nn.ModuleList([ 
                nn.Conv2d(self.chids[i], 1, ksz, 1, pad, bias=True) 
                for i in range(self.nlayers)])
        else: 
            self.attn = None 

        self.top = nn.Conv2d(chids[-1], cinp, ksz, 1, pad, bias=True) 

    def forward(self, hidden, target, attn_masks=None): 

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
            mask = [target.new(bsz, cinp, height, width).bernoulli_(1-self.dropout).div(1-self.dropout)]
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

                if self.attn is not None: 
                    if attn_masks is not None: 
                        a = attn_masks[step-1][i]
                        x = x*a.expand_as(x) 
                    else: 
                        a = nn.Sigmoid()(self.attn[i](x)) 
                        x = x*a.expand_as(x) 

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
    def __init__(self, cinp, chids, ksz, dropout=0.2, h_dropout=0.0, gate=False): 
        super(ConvEncDec, self).__init__() 

        self.dropout = dropout 
        self.h_dropout = h_dropout

        self.convEncoder = ConvEncoder(cinp, chids, ksz, dropout=dropout, h_dropout=h_dropout, gate=gate) 
        self.convReconstructor= ConvDecoder(cinp, chids, ksz, reverse=True, dropout=dropout, h_dropout=h_dropout, gate=gate) 
        self.convPredictor = ConvDecoder(cinp, chids, ksz, dropout=dropout, h_dropout=h_dropout, gate=gate) 
        
        # tying attention network
        for i in range(len(chids)):     
            self.convEncoder.layer_stack[i].i2h.weight = self.convPredictor.layer_stack[i].i2h.weight 
            self.convEncoder.layer_stack[i].i2h.bias = self.convPredictor.layer_stack[i].i2h.bias
            self.convEncoder.layer_stack[i].h2h.weight = self.convPredictor.layer_stack[i].h2h.weight 

            if gate is True: 
                for i in range(len(chids)): 
                    self.convEncoder.attn[i].weight = self.convPredictor.attn[i].weight 
                    self.convEncoder.attn[i].bias = self.convPredictor.attn[i].bias


    def forward(self, input, target, hidden=None):

        hidden, attns = self.convEncoder(input, hidden)

        reconstructed = self.convReconstructor(hidden, input, attns) 
        predicted = self.convPredictor(hidden, target) 
        output = torch.cat([reconstructed, predicted], 1) 

        return output, hidden
