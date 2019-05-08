# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:00:19 2019

@author: rober
"""

import numpy as np
import torch
import random


def batch_generator(X, Y, n_batches):  
    
    random.seed(123)
    
    batch_size = X.shape[0] // n_batches
    
    idx = list(X.index)
    random.shuffle(idx)    
    idx = idx[:n_batches*batch_size]
        
    for i in range(n_batches):            
        bi = np.random.choice(idx, batch_size, replace=False)
        X_batch = X.loc[bi]
        Y_batch = Y[bi]
        idx = [i for i in idx if i not in bi]
        yield (X_batch,Y_batch) 
        
        
def comb_error(output, target, sig2):
    output = (output + 0.0001)*0.999
    logErr1 = -1 * torch.log(output) * target[:,0].view(1,-1).t()
    logErr2 = -1 * torch.log(1 - output) * (1 - target[:,0].view(1,-1)).t()
    mseErr = torch.pow(target - output, 2) * (1 - target[:,0].view(1,-1)).t()
    return (1/output.shape[0]) * (torch.sum(logErr1,dim=0)[0] + torch.sum(logErr2,dim=0)[0] + 
                                                                        (1/(2*sig2))*torch.sum(mseErr,dim=0)[1])


def fit(X, X_val, Y, Y_val, net, optimizer, error, n_epochs, 
            n_batches, batch_to_avg, ep_to_check, clipping, PATH, device, verbose, min_val_loss = float('inf')):
    
    net = net.to(device)
    
    losses = []
    val_losses = []

    val_inputs = torch.FloatTensor(X_val.values)
    val_labels = torch.FloatTensor(Y_val)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)    
    
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
    
        running_loss = 0
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        counter = 0
            
        for batch_x, batch_y in batch_generator(X, Y, n_batches):  
            
            counter += 1 
                       
            # get the inputs
            inputs = torch.FloatTensor(batch_x.values)
            labels = torch.FloatTensor(batch_y)
            inputs, labels = inputs.to(device), labels.to(device)             
                
    
            # forward + backward + optimize
            outputs = net.forward(inputs)
            loss = error(outputs, labels)
                        
            loss.backward()            
            
            running_loss += loss.item()        
            
            if counter % batch_to_avg == 0:
                
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping) 
                
                optimizer.step()
                optimizer.zero_grad()         
                
                
        running_loss = running_loss/n_batches       
        
        with torch.no_grad():
            val_outputs = net.forward(val_inputs)
            val_loss = error(val_outputs, val_labels) 
        
        losses.append(running_loss)
        val_losses.append(val_loss.item())
        
        
        
        if verbose == 1:
            print('Epoch {0}: Training Loss: {1}, Validation Loss: {2}'\
                  .format(epoch+1, running_loss, val_loss.item()))
        
        
        
        if (epoch % ep_to_check == 0) and (epoch >= ep_to_check):
            
            mean_val_loss = np.mean(val_losses[-ep_to_check:])
            
              
            if mean_val_loss < min_val_loss:
                torch.save(net.state_dict(), PATH)
                if verbose == 1:
                    print('New Checkpoint Saved into PATH')
                min_val_loss = mean_val_loss
    
    return (losses, val_losses, min_val_loss)
    