import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import os
from math import pi
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

from model import *

EPSILON = Variable(torch.Tensor([1e-6]).cuda(),requires_grad=False)
PI = Variable(torch.Tensor([pi]).cuda(),requires_grad=False)

LR = 1e-4
EPOCH_VAE = 5
EPOCH_VAWGAN =11

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
        
class Trainer:
   
    def __init__(self):

        self.G = G().cuda()
        self.G.apply(weights_init)
        self.D = D().cuda()
        self.D.apply(weights_init)
        self.Encoder = Encoder().cuda()
        self.Encoder.apply(weights_init)
        self.batch_size = 256
    
    def load_data(self,x,y):
        self.source = x
        self.target = y          
    
    def circuit_loop(self,who_feature,who_label):
        
        z_mu,z_lv = self.Encoder(who_feature)
        z = GaussianSampleLayer(z_mu,z_lv)
        x_logit,x_feature = self.D(who_feature)     
        xh,xh_sig_logit = self.G(z,who_label)#[256,128]#[256,1]     
        xh_logit,xh_feature = self.D(xh)#xh_logit[256,1]
      
      
    
    
    
        return dict(
            z = z,
            z_mu = z_mu,
            z_lv = z_lv,
            xh= xh,
            xh_sig_logit = xh_sig_logit,
            x_logit = x_logit,
            x_feature = x_feature,
            xh_logit = xh_logit,
            xh_feature = xh_feature,
            
        )
  
    def train(self):
        
        gan_loss = 50000
        x_feature = torch.FloatTensor(-1, 1, 513, 1).cuda()#NHWC
        x_label = torch.FloatTensor(self.batch_size).cuda()
        y_feature = torch.FloatTensor(-1, 1, 513, 1).cuda()#NHWC
        y_label = torch.FloatTensor(self.batch_size).cuda()

        x_feature = Variable(x_feature)
        x_label = Variable(x_label,requires_grad=False)
        y_feature = Variable(y_feature)
        y_label = Variable(y_label,requires_grad=False)


        optimD = optim.RMSprop([{'params':self.D.parameters()}], lr=LR)
        optimG = optim.RMSprop([{'params':self.G.parameters()}], lr=LR)
        optimE = optim.RMSprop([{'params':self.Encoder.parameters()}], lr=LR)

        schedulerD = torch.optim.lr_scheduler.StepLR(optimD, step_size=10, gamma=0.1)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimG, step_size=10, gamma=0.1)
        schedulerE = torch.optim.lr_scheduler.StepLR(optimE, step_size=10, gamma=0.1)
        
        Data = DataLoader( 
            ConcatDataset(self.source,self.target),
            batch_size=self.batch_size ,shuffle = True,num_workers = 1)
     
        #print('N H W C')
       
        for epoch in range(EPOCH_VAE):
           
            #schedulerD.step()
            #schedulerG.step()
            #schedulerE.step()
            for index,(s_data,t_data) in enumerate(Data):
                
               
                #Source
                feature_1 = s_data[:,:513,:,:].permute(0,3,1,2)#NHWC ==> NCHW
                label_1 = s_data[:,-1,:,:].view(len(s_data))
               
                x_feature.data.resize_(feature_1.size())
                x_label.data.resize_(len(s_data))
                
                x_feature.data.copy_(feature_1)
                x_label.data.copy_(label_1)
              
              
                s = self.circuit_loop(x_feature,x_label) 
                
                #Target
                feature_2 = t_data[:,:513,:,:].permute(0,3,1,2)#NHWC ==> NCHW
                label_2 = t_data[:,-1,:,:].view(len(t_data))
               
                y_feature.data.resize_(feature_2.size())
                y_label.data.resize_(len(t_data))
                
                y_feature.data.copy_(feature_2)
                y_label.data.copy_(label_2)
              
                
                t = self.circuit_loop(y_feature,y_label) 
                
                #Source 2 Target
                s2t = self.circuit_loop(x_feature,y_label) 
        
                
                
               
                loss = dict()
                loss['conv_s2t'] = \
                      reconst_loss(t['x_logit'],s2t['xh_logit'])  
                loss['conv_s2t'] *= 100   
                     
                loss['KL(z)'] = \
                    torch.mean(
                        GaussianKLD(
                            s['z_mu'], s['z_lv'],
                            torch.zeros_like(s['z_mu']), torch.zeros_like(s['z_lv']))) + \
                    torch.mean(
                        GaussianKLD(
                            t['z_mu'], t['z_lv'],
                            torch.zeros_like(t['z_mu']), torch.zeros_like(t['z_lv'])))        
                loss['KL(z)'] /= 2.0  
                           
                    
                loss['Dis'] = \
                   torch.mean(
                        GaussianLogDensity(
                            x_feature.view(-1,513),
                            s['xh'].view(-1,513),
                            torch.zeros_like(x_feature.view(-1,513))))  + \
                   torch.mean(
                        GaussianLogDensity(
                            y_feature.view(-1,513),
                            t['xh'].view(-1,513),
                            torch.zeros_like(y_feature.view(-1,513))))                         
                loss['Dis'] /= - 2.0
                
                
                
                optimE.zero_grad()
                obj_Ez = loss['KL(z)'] + loss['Dis']
                obj_Ez.backward(retain_graph=True)
                optimE.step()
                
                optimG.zero_grad()
                obj_Gx = loss['Dis']
                obj_Gx.backward()
                optimG.step()
                
                
         
                
               
                
                
               
          
                print("Epoch:[%d|%d]\tIteration:[%d|%d]\tW: %.3f\tKL(Z): %.3f\tDis: %.3f" %(epoch+1,EPOCH_VAWGAN+EPOCH_VAE,index+1,len(Data),
                                                     loss['conv_s2t'],loss['KL(z)'],loss['Dis']  ))
           
    
        for epoch in range(EPOCH_VAWGAN):
           
            schedulerD.step()
            schedulerG.step()
            schedulerE.step()
            for index,(s_data,t_data) in enumerate(Data):       
                
                
                #Source
                feature_1 = s_data[:,:513,:,:].permute(0,3,1,2)#NHWC ==> NCHW
                label_1 = s_data[:,-1,:,:].view(len(s_data))
               
                x_feature.data.resize_(feature_1.size())
                x_label.data.resize_(len(s_data))
                
                x_feature.data.copy_(feature_1)
                x_label.data.copy_(label_1)
              
                
                #Target
                feature_2 = t_data[:,:513,:,:].permute(0,3,1,2)#NHWC ==> NCHW
                label_2 = t_data[:,-1,:,:].view(len(t_data))
               
                y_feature.data.resize_(feature_2.size())
                y_label.data.resize_(len(t_data))
                
                y_feature.data.copy_(feature_2)
                y_label.data.copy_(label_2)
              
                
                t = dict() 
                
                #Source 2 Target
                s2t = dict()
            
                loss = dict()    
                
                if (epoch+EPOCH_VAE == EPOCH_VAE and index < 25) or (index % 100 == 0):
                    D_Iter = 100
                else:
                    D_Iter = 10
                   
                for D_index in range(D_Iter):
                    for p in self.D.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    #Target result
                    optimD.zero_grad()
                    t = self.circuit_loop(y_feature,y_label) 
                    #Source 2 Target result
                    s2t = self.circuit_loop(x_feature,y_label) 
                
                    loss['conv_s2t'] = \
                          reconst_loss(t['x_logit'],s2t['xh_logit'])  
                    
                    loss['conv_s2t'] *= 100  
                    #print ("%.3f\t" %(loss['conv_s2t'])) 
                    #print(   loss )
                    obj_Dx = -0.01*loss['conv_s2t']
                    obj_Dx.backward(retain_graph=True)
                    optimD.step() 
                      
                
                #Source result
                s = self.circuit_loop(x_feature,x_label) 
                
                loss['KL(z)'] = \
                    torch.mean(
                        GaussianKLD(
                            s['z_mu'], s['z_lv'],
                            torch.zeros_like(s['z_mu']), torch.zeros_like(s['z_lv']))) + \
                    torch.mean(
                        GaussianKLD(
                            t['z_mu'], t['z_lv'],
                            torch.zeros_like(t['z_mu']), torch.zeros_like(t['z_lv'])))        
                loss['KL(z)'] /= 2.0  
                           
                    
                loss['Dis'] = \
                   torch.mean(
                        GaussianLogDensity(
                            x_feature.view(-1,513),
                            s['xh'].view(-1,513),
                            torch.zeros_like(x_feature.view(-1,513))))  + \
                   torch.mean(
                        GaussianLogDensity(
                            y_feature.view(-1,513),
                            t['xh'].view(-1,513),
                            torch.zeros_like(y_feature.view(-1,513))))                         
                loss['Dis'] /= - 2.0
                #print(   loss )
                
                #print ("%.3f\t" %(loss['conv_s2t']))
                
                optimE.zero_grad()
                obj_Ez = loss['KL(z)'] + loss['Dis']
                obj_Ez.backward(retain_graph=True)
                optimE.step()
                
                optimG.zero_grad()
                obj_Gx = loss['Dis'] +50  * loss['conv_s2t']
                obj_Gx.backward()
                optimG.step()
                print("Epoch:[%d|%d]\tIteration:[%d|%d]\t[D_loss: %.3f\tG_loss: %.3f\tE_loss: %.3f]\t[S2T: %.3f\tKL(z): %.3f\tDis: %.3f]" %(EPOCH_VAE+epoch+1,EPOCH_VAWGAN+EPOCH_VAE,index+1,len(Data),
                                                      -0.01*loss['conv_s2t'],loss['Dis']+ 50 * loss['conv_s2t'],loss['Dis']+loss['KL(z)'], loss['conv_s2t'], loss['KL(z)'], loss['Dis'] ))
            
                if( epoch == EPOCH_VAWGAN-1 and index == (len(Data)-2)):
                    print('================= store model ==================')
                    filename = './model/model_'+str(epoch+EPOCH_VAE+1)+'.pt'
                    if not os.path.exists(os.path.dirname(filename)):
                        try:
                            os.makedirs(os.path.dirname(filename))
                        except OSError as exc: # Guard against race condition
                            print('error')
                            pass
                              
                    torch.save(self,filename)
                    print('=================Finish store model ==================')
                    gan_loss=  obj_Gx 
                
                
       
                
          
           
    
      
      
            
          
      

def reconst_loss(x,xh):
    return torch.mean(x) - torch.mean(xh)
    
def GaussianSampleLayer(z_mu, z_lv):   
    std = torch.sqrt(torch.exp(z_lv))
    eps = torch.randn_like(std)
    return eps.mul(std).add_(z_mu)
 
 
def GaussianLogDensity(x, mu, log_var):
   
    c = torch.log(2. * PI)
    var = torch.exp(log_var)
    x_mu2 = torch.mul(x - mu,x - mu)  # [Issue] not sure the dim works or not?
    x_mu2_over_var = torch.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = torch.sum(log_prob, 1)   # keep_dims=True,
    return log_prob

def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
   
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    mu_diff_sq =  torch.mul(mu1 - mu2,mu1 - mu2)
    dimwise_kld = .5 * (
        (lv2 - lv1) + torch.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
    
    return torch.sum(dimwise_kld,1)    
  

    
    
