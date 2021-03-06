# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:00:19 2019

@author: rober
"""

import numpy as np
import torch
import random


def batch_generator(X, Y, n_batches):  
    
    random.seed(0)
    np.random.seed(0)
    
    batch_size = X.shape[0] // n_batches
    
    idx = list(range(X.shape[0]))
    random.shuffle(idx)    
    idx = idx[:n_batches*batch_size]
        
    for i in range(n_batches):            
        bi = np.random.choice(idx, batch_size, replace=False)
        X_batch = X[bi]
        Y_batch = Y[bi]
        idx = [i for i in idx if i not in bi]
        yield (X_batch,Y_batch)
        
        
def comb_error(output, target, sig2):
    output = (output + 0.0001)*0.999 #help avoid numerical errors
    p = target[:,0]
    
    logErr1 = torch.mul(torch.log(output[:,0]), p)
    logErr2 = torch.mul(torch.log(1 - output[:,0]) , (1 - p))
    mseErr = torch.mul(torch.pow(target[:,1] - output[:,1], 2), (1 - p))
    
    logErr1 = -1 * torch.sum(logErr1)
    logErr2 = -1 * torch.sum(logErr2)
    mseErr = torch.sum(mseErr)    
    
    return (1/output.shape[0]) * (logErr1 + logErr2 + (1/sig2)*mseErr) 


def mae_error(output, target):
    y = target[:,1]
    p = output[:,0]
    f2 = output[:,1]
    yhat = (1-p)*f2
    MAE = np.mean(np.absolute(y - yhat))
    return MAE    


def fmapper(x):
    y = x.copy()
    y[x<9] = 1
    y[(x>=9) & (x<53)] = 2
    y[(x>=53) & (x<172)] = 3
    y[x>=172] = 4  
    return y


def hrat_error(output, target):
    y = target[:,1]
    p = output[:,0]    
    
    f2 = output[:,1]
    yhat = (1-(p>0.5)*1)*f2
    
    yhat = fmapper(yhat)
    y = fmapper(y)    
    hr = sum((yhat==y)*1)/len(y)
    return (1-hr) 


def fit(X, X_val, Y, Y_val, net, optimizer, error, val_error, n_epochs, 
            n_batches, batch_to_avg, ep_to_check, clipping, PATH, device, verbose, min_val_loss = float('inf')):
    
    torch.manual_seed(0)
    net = net.to(device)
    
    losses = []
    val_losses = []

    val_inputs = torch.FloatTensor(X_val)
    val_labels = torch.FloatTensor(Y_val)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)    
    
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
    
        running_loss = 0
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        counter = 0
            
        for batch_x, batch_y in batch_generator(X, Y, n_batches):  
            
            counter += 1 
                       
            net.train()
            # get the inputs
            inputs = torch.FloatTensor(batch_x)
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
            net.eval()
            val_outputs = net.forward(val_inputs)
            
            val_outputs2, val_labels2 = val_outputs.cpu(), val_labels.cpu()
            val_outputs2, val_labels2 = val_outputs2.numpy(), val_labels2.numpy()
            
            val_loss = val_error(val_outputs2, val_labels2)
           
        
        losses.append(running_loss)
        val_losses.append(val_loss)        
        
        
        if verbose == 1:
            print('Epoch {0}: Training Loss: {1}, Validation Loss: {2}'\
                  .format(epoch+1, running_loss, val_loss))
        
        
        
        if (epoch % ep_to_check == 0) and (epoch >= ep_to_check):
            
            mean_val_loss = np.mean(val_losses[-ep_to_check:])
        
            
              
            if mean_val_loss < min_val_loss:
        
           
                torch.save(net.state_dict(), PATH)
                if verbose == 1:
                    print('New Checkpoint Saved into PATH')
                min_val_loss = mean_val_loss
    
    return (losses, val_losses, min_val_loss)
    