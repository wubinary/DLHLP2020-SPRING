import torch
import torch.nn as nn
import numpy as np
import argparse
import soundfile as sf
import os
from analyzer import Tanhize, read_whole_features,pw2wav
from torch.autograd import Variable
from datetime import datetime
parser = argparse.ArgumentParser(description='Voice Convert.py')
parser.add_argument('--corpus_name', default='vcc2016', help='Corpus name')
parser.add_argument('--src', default='SF1', help='source speaker [SF1 - SM2]')
parser.add_argument('--trg', default='TM3', help='target speaker [SF1 - TM3]')
parser.add_argument('--output_dir', default='./logdir', help='root of output dir')
parser.add_argument('--model_name', default='./model/vawgan.pt', help='load ./model/[vawgan.pt]')
parser.add_argument('--file_pattern', default='./dataset/vcc2016/bin/Testing Set/{}/*.bin', help='file pattern')
parser.add_argument('--speaker_list', default='./etc/speakers.tsv', help='Speaker list (one speaker per line)')

args = parser.parse_args()
def nh_to_nchw(x):
  
    return x.reshape(-1,1,513,1)
    
def convert_f0(f0, src, trg):
    print(f0)
    print(np.fromfile(os.path.join('./etc', '{}.npf'.format(src)), np.float32))
    print(np.fromfile(os.path.join('./etc', '{}.npf'.format(trg)), np.float32))
    mu_s, std_s = np.fromfile(os.path.join('./etc', '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join('./etc', '{}.npf'.format(trg)), np.float32)
    print('//')
    lf0 = np.where(f0 > 1., np.log(f0), f0)
    print(lf0)
    print('//')
    lf0 = np.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    print('//')
    lf0 = np.where(lf0 > 1., np.exp(lf0), lf0)
    return lf0
def get_default_output(logdir_root):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, 'output', STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))        
    return logdir
def make_output_wav_name(output_dir, filename):
    print(filename)
    print(type(filename))
    basename = filename
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    # print('Processing {}'.format(basename))        
    return os.path.join(
        output_dir, 
        '{}-{}-{}.wav'.format(args.src, args.trg, basename)
    )
def main():
     

    model = torch.load('./model/'+str(args.model_name)+'.pt')
    FS = 16000
    SPEAKERS = list()
    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]
        
    
    
    normalizer = Tanhize(
        xmax=np.fromfile('./etc/{}_xmax.npf'.format(args.corpus_name)),
        xmin=np.fromfile('./etc/{}_xmin.npf'.format(args.corpus_name)),
    )
    
    total_sp_speaker = []
    total_speaker = []
    
    total_features = read_whole_features(args.file_pattern.format(args.src))
    for features in total_features:
        
        x = normalizer.forward_process(features['sp'])
        x = nh_to_nchw(x)
        y_s = features['speaker']
        print('????',SPEAKERS.index(args.trg))
        
        
        #y_t_id = tf.placeholder(dtype=tf.int64, shape=[1,])
        #y_t = y_t_id * torch.ones(shape=[x.shape[0],], dtype=torch.int64)
        #print(y_t)
        x = Variable(torch.FloatTensor(x).cuda(),requires_grad=False)
       
        y_t = torch.ones((x.shape[0])).view(-1,1)*(SPEAKERS.index(args.trg))
   
        z,_ = model.Encoder(x)
        x_t,_ = model.G(z, y_t)  # NOTE: the API yields NHWC format
        x_t = torch.squeeze(x_t)
        print('x_t.shape',x_t.shape)
        x_t = normalizer.backward_process(x_t)
        print('backward_process.finish')
        
        x_s,_ = model.G(z, y_s)
        x_s = torch.squeeze(x_s)
        x_s = normalizer.backward_process(x_s)
        
        f0_s = features['f0']
        print(f0_s.shape)
        f0_t = convert_f0(f0_s, args.src, args.trg)
        
        output_dir = get_default_output(args.output_dir)
        features['sp'] = x_t.cpu().data.numpy()
        features['f0'] = f0_t
        print('=-=-=-=-=-=')
        y = pw2wav(features)
        
        oFilename = make_output_wav_name(output_dir, features['filename'])
        print('\rProcessing {}'.format(oFilename), end='')
        print(oFilename)
       
        if not os.path.exists(os.path.dirname(oFilename)):
            try:
                os.makedirs(os.path.dirname(oFilename))
            except OSError as exc: # Guard against race condition
                print('error')
                pass
                  
       
        sf.write(oFilename, y, FS)
        print('2: ',features['sp'].shape)
        print('3: ',features['f0'].shape)
        
    print('==finish==')
if __name__ == '__main__':
    main()
