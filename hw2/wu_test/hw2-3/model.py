import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

########################################################################################
#######################################  Encoder #######################################

def pad_layer(inp, layer, is_2d=False):
    if type(layer.kernel_size) == tuple:
        kernel_size = layer.kernel_size[0]
    else:
        kernel_size = layer.kernel_size
    if not is_2d:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2)
    else:
        if kernel_size % 2 == 0:
            pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
        else:
            pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out

def linear(inp, layer):
    batch_size = inp.size(0)
    hidden_dim = inp.size(1)
    seq_len = inp.size(2)
    inp_permuted = inp.permute(0, 2, 1)
    inp_expand = inp_permuted.contiguous().view(batch_size*seq_len, hidden_dim)
    out_expand = layer(inp_expand)
    out_permuted = out_expand.view(batch_size, seq_len, out_expand.size(1))
    out = out_permuted.permute(0, 2, 1)
    return out

def RNN(inp, layer):
    inp_permuted = inp.permute(2, 0, 1)
    state_mul = (int(layer.bidirectional) + 1) * layer.num_layers
    zero_state = Variable(torch.zeros(state_mul, inp.size(0), layer.hidden_size))
    zero_state = zero_state.cuda() if torch.cuda.is_available() else zero_state
    out_permuted, _ = layer(inp_permuted, zero_state)
    out_rnn = out_permuted.permute(1, 2, 0)
    return out_rnn

class Encoder(nn.Module):
    def __init__(self, c_in=513, c_h1=150, c_h2=150, c_h3=128, ns=0.2, dp=0.5):
        super(Encoder, self).__init__()
        self.ns = ns
        self.conv1s = nn.ModuleList(
                [nn.Conv1d(c_in, c_h1, kernel_size=k) for k in range(1, 8)]
            )
        self.conv2 = nn.Conv1d(len(self.conv1s)*c_h1 + c_in, c_h2, kernel_size=1)
        self.conv3 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        self.conv7 = nn.Conv1d(c_h2, c_h2, kernel_size=5)
        self.conv8 = nn.Conv1d(c_h2, c_h2, kernel_size=5, stride=2)
        # mean
        self.dense1 = nn.Linear(c_h2, c_h2)
        self.dense2 = nn.Linear(c_h2, c_h2)
        self.dense3 = nn.Linear(c_h2, c_h2)
        self.dense4 = nn.Linear(c_h2, c_h2)
        self.RNN = nn.GRU(input_size=c_h2, hidden_size=c_h3, num_layers=1, bidirectional=True)
        self.linear0 = nn.Linear(c_h2 + 2*c_h3, c_h2)
        # std
        self.linear1 = nn.Linear(c_h2 + 2*c_h3, c_h2)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h2)
        self.ins_norm2 = nn.InstanceNorm1d(c_h2)
        self.ins_norm3 = nn.InstanceNorm1d(c_h2)
        self.ins_norm4 = nn.InstanceNorm1d(c_h2)
        self.ins_norm5 = nn.InstanceNorm1d(c_h2)
        self.ins_norm6 = nn.InstanceNorm1d(c_h2)
        # dropout layer
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.drop5 = nn.Dropout(p=dp)
        self.drop6 = nn.Dropout(p=dp)

    def conv_block(self, x, conv_layers, norm_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            x_pad = F.pad(x, pad=(0, x.size(2) % 2), mode='reflect')
            x_down = F.avg_pool1d(x_pad, kernel_size=2)
            out = x_down + out 
        return out

    def dense_block(self, x, layers, norm_layers, res=True):
        out = x
        for layer in layers:
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in norm_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        outs = []
        for l in self.conv1s:
            out = pad_layer(x, l)
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = self.conv_block(out, [self.conv2], [self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2])
        out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3])
        out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4])
        # dense layer
        out = self.dense_block(out, [self.dense1, self.dense2], [self.ins_norm5, self.drop5], res=True)
        out = self.dense_block(out, [self.dense3, self.dense4], [self.ins_norm6, self.drop6], res=True)
        out_rnn = RNN(out, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        mean = linear(out, self.linear0)
        #mean = F.relu(mean)#, negative_slope=self.ns)
        
        std = linear(out, self.linear1)
        std = F.relu(std)#, negative_slope=self.ns)
        return mean, std
    
########################################################################################
#######################################  Decoder #######################################

def pixel_shuffle_1d(inp, upscale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= upscale_factor
    
    out_width = in_width * upscale_factor
    inp_view = inp.contiguous().view(batch_size, channels, upscale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out
    
def upsample(x, scale_factor=2):
    x_up = F.upsample(x, scale_factor=2, mode='nearest')
    return x_up

def append_emb(emb, expand_size, output):
    emb = emb.unsqueeze(dim=2)
    emb_expand = emb.expand(emb.size(0), emb.size(1), expand_size)
    output = torch.cat([output, emb_expand], dim=1)
    return output

class Decoder(nn.Module):
    def __init__(self, c_in=150, c_out=513, c_h=150, c_a=2, emb_size=128, ns=0.2):
        super(Decoder, self).__init__()
        self.ns = ns
        self.conv1 = nn.Conv1d(c_in, 2*c_h, kernel_size=3)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv3 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.conv5 = nn.Conv1d(c_h, 2*c_h, kernel_size=3)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=3)
        self.dense1 = nn.Linear(c_h, c_h)
        self.dense2 = nn.Linear(c_h, c_h)
        self.dense3 = nn.Linear(c_h, c_h)
        self.dense4 = nn.Linear(c_h, c_h)
        self.RNN = nn.GRU(input_size=c_h, hidden_size=c_h//2, num_layers=1, bidirectional=True)
        self.dense5 = nn.Linear(2*c_h + c_h, c_h)
        self.linear = nn.Linear(c_h, c_out)
        # normalization layer
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h)
        self.ins_norm5 = nn.InstanceNorm1d(c_h)
        # embedding layer
        self.emb1 = nn.Embedding(c_a, c_h)
        self.emb2 = nn.Embedding(c_a, c_h)
        self.emb3 = nn.Embedding(c_a, c_h)
        self.emb4 = nn.Embedding(c_a, c_h)
        self.emb5 = nn.Embedding(c_a, c_h)

    def conv_block(self, x, conv_layers, norm_layer, emb, res=True):
        # first layer
        x_add = x + emb.view(emb.size(0), emb.size(1), 1)
        out = pad_layer(x_add, conv_layers[0])
        out = F.leaky_relu(out, negative_slope=self.ns)
        # upsample by pixelshuffle
        out = pixel_shuffle_1d(out, upscale_factor=2)
        out = out + emb.view(emb.size(0), emb.size(1), 1)
        out = pad_layer(out, conv_layers[1])
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            x_up = upsample(x, scale_factor=2)
            out = out + x_up
        return out

    def dense_block(self, x, emb, layers, norm_layer, res=True):
        out = x
        for layer in layers:
            out = out + emb.view(emb.size(0), emb.size(1), 1)
            out = linear(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        out = norm_layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x, c):
        # conv layer
        out = self.conv_block(x, [self.conv1, self.conv2], self.ins_norm1, self.emb1(c), res=True )
        out = self.conv_block(out, [self.conv3, self.conv4], self.ins_norm2, self.emb2(c), res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], self.ins_norm3, self.emb3(c), res=True)
        # dense layer
        out = self.dense_block(out, self.emb4(c), [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, self.emb4(c), [self.dense3, self.dense4], self.ins_norm5, res=True)
        emb = self.emb5(c)
        out_add = out + emb.view(emb.size(0), emb.size(1), 1)
        # rnn layer
        out_rnn = RNN(out_add, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = append_emb(self.emb5(c), out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        #out = torch.tanh(out)
        return out
    
    def interpolate(self, x, c1, c2):
        # conv layer
        #print(self.emb1(c1).shape)
        #input('')
        out = self.conv_block(x, [self.conv1, self.conv2], self.ins_norm1, (self.emb1(c1)+self.emb1(c2))/2, res=True )
        out = self.conv_block(out, [self.conv3, self.conv4], self.ins_norm2, (self.emb2(c1)+self.emb2(c2))/2, res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], self.ins_norm3, (self.emb3(c1)+self.emb3(c2))/2, res=True)
        # dense layer
        out = self.dense_block(out, (self.emb4(c1)+self.emb4(c2))/2, [self.dense1, self.dense2], self.ins_norm4, res=True)
        out = self.dense_block(out, (self.emb4(c1)+self.emb4(c2))/2, [self.dense3, self.dense4], self.ins_norm5, res=True)
        emb = (self.emb5(c1)+self.emb5(c2))/2
        out_add = out + emb.view(emb.size(0), emb.size(1), 1)
        # rnn layer
        out_rnn = RNN(out_add, self.RNN)
        out = torch.cat([out, out_rnn], dim=1)
        out = append_emb((self.emb5(c1)+self.emb5(c2))/2, out.size(2), out)
        out = linear(out, self.dense5)
        out = F.leaky_relu(out, negative_slope=self.ns)
        out = linear(out, self.linear)
        #out = torch.tanh(out)
        return out
    
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def reparameterized(self, mean, std):
        return mean + torch.randn_like(std)*torch.exp(0.5*std)
    
    def forward(self, x, c):
        mean, std = self.encoder(x)
        if self.training:
            z = self.reparameterized(mean, std)
        else:
            z = mean 
        out = self.decoder(z, c)
        return mean, out
    
    def interpolate(self, x, c1, c2):
        mean, std = self.encoder(x)
        z = mean 
        out = self.decoder.interpolate(z, c1, c2)
        return out
    

    
