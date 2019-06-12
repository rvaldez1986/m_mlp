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
        
        
def comb_error(output, target, sig1, sig2):
    output = (output + 0.0001)*0.999
    p0 = target[:,0]
    p1 = target[:,1]
    p2 = target[:,2]
    
    logErr0 = torch.mul(torch.log(output[:,0]), p0)
    logErr1 = torch.mul(torch.log(output[:,1]), p1)
    logErr2 = torch.mul(torch.log(output[:,2]), p2)
    mseErr1 = torch.mul(torch.pow(target[:,2+1] - output[:,2+1], 2), p1)
    mseErr2 = torch.mul(torch.pow(target[:,2+2] - output[:,2+2], 2), p2)   
    
    logErr0 = -1 * torch.sum(logErr0)
    logErr1 = -1 * torch.sum(logErr1)
    logErr2 = -1 * torch.sum(logErr2)
    mseErr1 = torch.sum(mseErr1)    
    mseErr2 = torch.sum(mseErr2)      
    
    return (1/output.shape[0]) * (logErr0 + logErr1 + logErr2 + (1/sig1)*mseErr1 + (1/sig2)*mseErr2)  


def w_comb_error(output, target, sig1, sig2, w0, w1, w2):
    output = (output + 0.0001)*0.999
    p0 = target[:,0]
    p1 = target[:,1]
    p2 = target[:,2]
    
    logErr0 = torch.mul(torch.log(output[:,0]), p0)
    logErr1 = torch.mul(torch.log(output[:,1]), p1)
    logErr2 = torch.mul(torch.log(output[:,2]), p2)
    mseErr1 = torch.mul(torch.pow(target[:,2+1] - output[:,2+1], 2), p1)
    mseErr2 = torch.mul(torch.pow(target[:,2+2] - output[:,2+2], 2), p2)   
    
    logErr0 = -1 * torch.sum(logErr0)
    logErr1 = -1 * torch.sum(logErr1)
    logErr2 = -1 * torch.sum(logErr2)
    mseErr1 = torch.sum(mseErr1)    
    mseErr2 = torch.sum(mseErr2)      
    
    return (1/output.shape[0]) * (w0*logErr0 + w1*logErr1 + w2*logErr2 + ((w1**2)/sig1)*mseErr1 + ((w2**2)/sig2)*mseErr2)    
    


def mae_error(output, target):  
    y = target[:,2+1] + target[:,2+2]  
    
    p1 = output[:,1]
    p2 = output[:,2]
    e1 = output[:,2+1]
    e2 = output[:,2+2]    
    yhat = p1 * e1 + p2 * e2
    
    MAE = np.mean(np.absolute(y - yhat))
    return MAE 


def w_mae_error(output, target, w0, w1, w2):  
    
    cut0 = 0
    cut1 = 1146    
    
    y = target[:,2+1] + target[:,2+2]  
    
    p1 = output[:,1]
    p2 = output[:,2]
    e1 = output[:,2+1]
    e2 = output[:,2+2]  
    
    yhat = p1 * e1 + p2 * e2    
    
    y0 = y[y==cut0]
    yh0 = yhat[y==cut0]
    y1 = y[(y>cut0) & (y<=cut1)]
    yh1 = yhat[(y>cut0) & (y<=cut1)]
    y2 = y[y>cut1]
    yh2 = yhat[y>cut1]
    
    n0 = len(y0)
    n1 = len(y1)
    n2 = len(y2)
 
    m0 = np.sum(np.abs(yh0 - y0))
    m1 = np.sum(np.abs(yh1 - y1))
    m2 = np.sum(np.abs(yh2 - y2))
    
    wt = n0*w0 + n1*w1 + n2*w2
    
    wMAE = (1/wt)*(w0*m0 + w1*m1 + w2*m2)
    return wMAE  





def class_error(output, target):  
    p0 = target[:,0]
    p1 = target[:,1]
    p2 = target[:,2]
    
    logErr0 = -1 * np.sum(np.log(output[:,0]) * p0)
    logErr1 = -1 * np.sum(np.log(output[:,1]) * p1)
    logErr2 = -1 * np.sum(np.log(output[:,2]) * p2)
    
    return (1/output.shape[0]) * (logErr0 + logErr1 + logErr2)

   
def loss_error(output, target, sig1, sig2):  
    p1 = target[:,1]
    p2 = target[:,2]
    
    mseErr1 = np.sum(((target[:,2+1] - output[:,2+1])**2) * p1 )   
    mseErr2 = np.sum(((target[:,2+2] - output[:,2+2])**2) * p2 ) 
    
    return (1/output.shape[0]) * ((1/sig1)*mseErr1  + (1/sig2)*mseErr2)
    

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


def fit(X, X_val, Y, Y_val, net, optimizer, error, val_error, class_error, loss_error, n_epochs, 
            n_batches, batch_to_avg, ep_to_check, clipping, PATH, device, verbose, min_val_loss = float('inf')):
    
    torch.manual_seed(0)
    net = net.to(device)
    
    losses = []
    val_losses = []
    class_losses = []
    l_losses = []

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
            
            class_loss = class_error(val_outputs2, val_labels2)
            l_loss = loss_error(val_outputs2, val_labels2)
            
            
           
        
        losses.append(running_loss)
        val_losses.append(val_loss)     
        class_losses.append(class_loss)
        l_losses.append(l_loss)
        
        
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
    
    return (losses, val_losses, min_val_loss, class_losses, l_losses)
    