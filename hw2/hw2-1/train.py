from model import cc, Encoder, Decoder, Autoencoder

import torch 
import torch.nn as nn

def _run_train(autoencoder, criterion, opt, dataloader):
        
    autoencoder.train()
    
    total_loss = 0
    for index, (person, spectrogram) in enumerate(dataloader):
        b = person.shape[0]
        
        opt.zero_grad()
    
        spectrogram = spectrogram.permute(0,2,1).cuda()
        person = person.cuda()
        
        latent, output = autoencoder(spectrogram, person)
        latent2 = autoencoder.encoder(output)
        
        loss = 0
        loss += criterion(spectrogram, output*spectrogram/spectrogram) #SI-SDR reconstruct loss
        loss += 0.1*criterion(latent, latent2) # content loss
        loss.backward()

        opt.step()
        
        total_loss += loss.item()*b
        print("\t [{}/{}] train loss:{:.4f}".format(index+1,
                                              len(dataloader),
                                              loss.item()), 
                                          end='  \r')
        
    return total_loss

def _run_eval(autoencoder, criterion, dataloader):
      
    autoencoder.eval()
    
    with torch.no_grad():
        total_loss = 0
        for index, (person, spectrogram) in enumerate(dataloader):
            b = person.shape[0]

            spectrogram = spectrogram.permute(0,2,1).cuda()
            person = person.cuda()
        
            latent, output = autoencoder(spectrogram, person)
            latent2 = autoencoder.encoder(output)

            loss = 0
            loss += criterion(spectrogram, output*spectrogram/spectrogram) #SI-SDR reconstruct loss
            loss += 0.1*criterion(latent, latent2) # content loss

            total_loss += loss.item()*b
            print("\t [{}/{}] valid loss:{:.4f}".format(index+1,
                                                  len(dataloader),
                                                  loss.item()), 
                                              end='  \r')                    
    return total_loss

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(87)
    torch.cuda.manual_seed(87)
    
    autoencoder = cc(Autoencoder())
    
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    
    best_loss = 1e100
    for epoch in range(args.epoch):
        print(f' Epoch {epoch}')
        
        loss = _run_train(autoencoder, criterion, opt, train_dataloader)
        print('\t [Info] Avg training loss:{:.5f}'.format(loss/len(train_dataloader.dataset)))

        loss = _run_eval(autoencoder, criterion, valid_dataloader)
        print('\t [Info] Avg valid loss:{:.5f}'.format(loss/len(valid_dataloader.dataset)))
        
        if True or loss < best_loss:
            best_loss = loss
            save_path = "{}/epoch_{}_loss_{:.4f}".format(args.save_path,epoch,loss/len(valid_dataloader.dataset))
            torch.save({'state_dict': autoencoder.state_dict()},
                        f"{save_path}_autoencoder_.pt")
            print(f'\t [Info] save weights at {save_path}')
        print('-----------------------------------------------')

        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] / 1

