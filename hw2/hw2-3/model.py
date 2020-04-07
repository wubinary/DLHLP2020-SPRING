import torch.nn as nn
from util.function import GaussianSampleLayer
import torch
from torch.autograd import Variable
    
    
        
class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        
        self.main = nn.Sequential(
          nn.Conv2d(1,16,(7,1),padding = (3,0),stride = (3,1),bias =False),
          #nn.BatchNorm2d(16),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(16,32,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(32),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(32,64,(115,1),padding = (57,0),stride = (3,1),bias =False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.02),\
        )
        self.fc = nn.Linear(1216,1,bias = True)

    def forward(self, x):
        #print('=====================D')
        #print(x.shape)
        h = x.view(-1,513)#原x = (256 * 512)
        #print('h: ',h.shape)
        output = self.main(x)
        #print(output.shape)
        output=output.view(-1, 1216)#19*64
        x = self.fc(output)#等於機率
        #print('=====================D_Finish x = ',x.shape)
        return x, h
    
    
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
          nn.Conv2d(1,16,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(16),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(16,32,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(32),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(32,64,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(64,128,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.02),\
          nn.Conv2d(128,256,(7,1),padding = (3,0),stride = (3,1),bias = False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.02),\
        )
        self.fc_mu = nn.Linear(768, 128,bias = True)
        self.fc_lv = nn.Linear(768, 128,bias = True)
   
   
    def forward(self, x):
        #print('call==================================================')
        #print(x.shape)
        #print(type(x))
        output = self.main(x)
        #print(output.shape)
        #print('?????????????????????????????')
        output=output.view(-1, 768)#3*256
        z_mu = self.fc_mu(output)
        z_lv = self.fc_lv(output)
        return z_mu, z_lv




class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        
        self.Embedding = nn.Linear(10,128,bias = False)
        
        self.fc1 = nn.Linear(128, 171, bias = True)
        self.fc2 = nn.Linear(128, 171, bias = True)
        self.LR =  nn.LeakyReLU(0.02)
        
        
        
        self.fc = nn.Sequential(
            nn.Linear(171,19*1*81,bias = True),
            nn.BatchNorm1d(19*1*81),\
            nn.LeakyReLU(0.02)
        )
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(81,32,(9,1),padding = (3,0),stride = (3,1),bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),\
            nn.ConvTranspose2d(32,16,(7,1),padding = (2,0),stride = (3,1),bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),\
            nn.ConvTranspose2d(16,8,(7,1),padding = (2,0),stride = (3,1),bias = False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),\
            nn.ConvTranspose2d(8,1,(1025,1),padding = (512,0),stride = (1,1),bias = False),
        )
        self.Tanh = nn.Tanh()

    def forward(self, z,y):
     
        person = torch.zeros(y.shape[0],10)
        for i in range(y.shape[0]):
            for j in range(10):
                if(j==y[i]):
                    person[i][j] = 1
                    break  
        #print('---------------: ')
        #print(person.shape)
        #print(person)
        who = Variable(person.cuda(),requires_grad=False)
 
        output = self.Embedding(who)
        
        x = 0
        _z = self.fc1(z)
        x += _z
        _y = self.fc2(output)
        x += _y
        x = self.LR(x)
       
        z = self.fc(x)
      
        z = z.view(-1,81,19,1)
    
        x = self.main(z)
        
        logit = x
        x = self.Tanh(x)
    
        return x,logit
    




def weights_init(m):
    classname = m.__class__.__name__
    print(m)
   
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
        #m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        #nn.init.xavier_normal_(m.weight.data)
        #m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        torch.nn.init.uniform(m.weight.data,-1, 1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        #m.weight.data.normal_(0.0, 0.02)
        #m.bias.data.fill_(0)
