# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:00:19 2019

@author: rober
"""

import numpy as np
import torch


def batch_generator(X, Y, batch_size):
        
    n_batches = X.shape[0] // batch_size
    idx = list(X.index)[:n_batches*batch_size]
        
    for i in range(n_batches):            
        bi = np.random.choice(idx, batch_size, replace=False)
        X_batch = X.loc[bi]
        Y_batch = Y[bi]
        idx = [i for i in idx if i not in bi]
        yield (X_batch,Y_batch) 



def fit(X, X_val, Y, Y_val, net, optimizer, error, n_epochs, 
            batch_size, iter_to_avg, lr, clipping, PATH, device, verbose):
    
    net = net.to(device)
    
    losses = []
    val_losses = []

    val_inputs = torch.FloatTensor(X_val.values)
    val_labels = torch.FloatTensor(Y_val)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
    min_val_loss = float('inf')
    
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
    
        running_loss = 0
        
        for _ in range(iter_to_avg):
            
            for batch_x, batch_y in batch_generator(X, Y, batch_size):            
                       
                # get the inputs
                inputs = torch.FloatTensor(batch_x.values)
                labels = torch.FloatTensor(batch_y)
                inputs, labels = inputs.to(device), labels.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net.forward(inputs)
                loss = error(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping)
                
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
        
        val_outputs = net.forward(val_inputs)
        val_loss = error(val_outputs, val_labels)  
        
        if verbose == 1:
            print('Epoch {0}: Training Loss: {1}, Validation Loss: {2}'\
                  .format(epoch+1, running_loss/iter_to_avg, val_loss.item()))
        losses.append(running_loss/iter_to_avg)
        val_losses.append(val_loss.item())
        
        if val_loss < min_val_loss:
            torch.save(net.state_dict(), PATH)
            if verbose == 1:
                print('New Checkpoint Saved into PATH')
            min_val_loss = val_loss
    
    
    