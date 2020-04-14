from model import cc, CVAE, Discriminator

import torch, random
import torch.nn as nn
    
STOP = float("inf")

def _run_train_autoencoder(model, criterion, optimizer, dataloader):
    
    autoencoder = model['autoen']
    autoencoder.train()
    
    opt = optimizer['autoen_adam']
    
    total_loss, count = 0, 1e-5
    for index, (person, _, spectrogram) in enumerate(dataloader):
        if index>STOP:
            break
        b = person.shape[0]      
        
        if random.random()<0.75:

            spectrogram = spectrogram.permute(0,2,1).cuda() #[b,f,t]
            person = person.cuda()

            output = autoencoder(spectrogram, person)

            opt.zero_grad()
            
            loss = criterion(spectrogram, output*spectrogram/spectrogram) #SI-SDR loss
            loss.backward()

            opt.step()

            count += b
            total_loss += loss.item()*b
            print("\t [{}/{}] training autoencoder, reconstruct loss:{:.4f}".format(index+1,
                                                                    len(dataloader),
                                                                    loss.item()), 
                                                                 end='  \r')
    return total_loss/count

def _run_train_discriminator(model, criterion, optimizer, dataloader):
    
    autoencoder = model['autoen']
    discriminator = model['discri']
    autoencoder.train()
    discriminator.train()
    
    opt = optimizer['discri_sgd']
    
    total_loss, count = 0, 1e-5
    for index, (person_a, person_b, spectrogram) in enumerate(dataloader):
        if index>STOP:
            break
        b = person_a.shape[0]      
        
        if True or random.random()<0.85:

            spectrogram = spectrogram.permute(0,2,1).cuda() #[b,f,t]
            person_a = person_a.cuda()
            person_b = person_b.cuda()

            spectrogram_a = autoencoder(spectrogram, person_a)
            spectrogram_b = autoencoder(spectrogram, person_b)
            
            output_r = discriminator(spectrogram)
            output_a = discriminator(spectrogram_a)
            output_b = discriminator(spectrogram_b)
            
            opt.zero_grad()

            loss = - criterion(output_a, output_r) - \
                    0.1*criterion(output_b, output_r)
            loss.backward()

            opt.step()

            count += b
            total_loss += loss.item()*b
            print("\t [{}/{}] training discriminator, loss:{:.4f}".format(index+1,
                                                                      len(dataloader),
                                                                      loss.item()), 
                                                                  end='  \r')
    return total_loss/count

def _run_train_adversarial(model, criterion, optimizer, dataloader):
        
    autoencoder = model['autoen']
    discriminator = model['discri']
    autoencoder.train()
    discriminator.train()
    
    opt_auto = optimizer['autoen_adam']
    opt_disc = optimizer['discri_sgd']
    
    total_auto_loss, total_disc_loss = 0, 0
    for index, (person_a, person_b, spectrogram) in enumerate(dataloader):
        #if index>STOP:
            #break
        b = person_a.shape[0]      
        
        spectrogram = spectrogram.permute(0,2,1).cuda() #[b,f,t]
        person_a = person_a.cuda()
        person_b = person_b.cuda()
        
        if index%2500<1500 or index==0: ## update autoencoder
            spectrogram_a = autoencoder(spectrogram, person_a)

            output_r = discriminator(spectrogram)
            output_a = discriminator(spectrogram_a)

            opt_auto.zero_grad()

            loss_auto = criterion(spectrogram_a*spectrogram/spectrogram, spectrogram) + \
                        0.05*criterion(output_a, output_r)
            loss_auto.backward()

            opt_auto.step()
        
        if index%2500>1500 or index==0: ## update discriminator
            spectrogram_a = autoencoder(spectrogram, person_a)
            spectrogram_b = autoencoder(spectrogram, person_b)

            output_r = discriminator(spectrogram)
            output_a = discriminator(spectrogram_a)
            output_b = discriminator(spectrogram_b)

            opt_disc.zero_grad()

            loss_disc = - criterion(output_a, output_r) - \
                        0.01*criterion(output_b, output_r)
            loss_disc.backward()

            opt_disc.step()
        
        total_auto_loss += loss_auto.item()*b
        total_disc_loss += loss_disc.item()*b
        print("\t [{}/{}] train loss, autoen:{:.4f} discrim:{:.4f}".format(index+1,
                                                                      len(dataloader),
                                                                      loss_auto.item(),
                                                                      loss_disc.item()), 
                                                                  end='                 \r')                
    return total_auto_loss/len(dataloader.dataset), total_disc_loss/len(dataloader.dataset)
    
def _run_eval(model, criterion, dataloader):
      
    autoencoder = model['autoen']
    discriminator = model['discri']
    autoencoder.eval()
    discriminator.eval()
    
    with torch.no_grad():
        total_auto_loss, total_disc_loss = 0, 0
        for index, (person_a, person_b, spectrogram) in enumerate(dataloader):
            b = person_a.shape[0]

            spectrogram = spectrogram.permute(0,2,1).cuda()
            person_a = person_a.cuda()
            person_b = person_b.cuda()
            
            spectrogram_a = autoencoder(spectrogram, person_a)
            spectrogram_b = autoencoder(spectrogram, person_b)

            output_r = discriminator(spectrogram)
            output_a = discriminator(spectrogram_a)
            output_b = discriminator(spectrogram_b)
            
            loss_auto = criterion(spectrogram_a*spectrogram/spectrogram, spectrogram)# - \
                        #criterion(output_a, output_r) + \
                        #criterion(output_b, output_r)
            loss_disc = - criterion(output_a, output_r) - 0.01*criterion(output_b, output_r)

            total_auto_loss += loss_auto.item()*b
            total_disc_loss += loss_disc.item()*b
            print("\t [{}/{}] valid loss, autoen reconstruct:{:.4f}, discrim:{:.4f}".format(index+1,
                                                                              len(dataloader),
                                                                              loss_auto.item(),
                                                                              loss_disc.item()), 
                                                                          end='  \r')
    return total_auto_loss/len(dataloader.dataset), total_disc_loss/len(dataloader.dataset)

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(87)
    torch.cuda.manual_seed(87)
    
    model = {'autoen':cc(CVAE()), 
             'discri':cc(Discriminator())}
    autoencoder, discriminator = model['autoen'], model['discri']
    
    criterion = torch.nn.MSELoss()
    
    optimizer = {'autoen_adam':torch.optim.Adam(autoencoder.parameters(), lr=args.lr),
                 'autoen_sgd':torch.optim.SGD(autoencoder.parameters(), lr=args.lr),
                 'discri_sgd':torch.optim.SGD(discriminator.parameters(), lr=args.lr)}
    
    ##########################################################################
    ##########             [Stage 1] train autoencoder              ##########
    
    ##round1
    avg_loss = _run_train_autoencoder(model, criterion, optimizer, train_dataloader)
    ##round2
    avg_loss = _run_train_autoencoder(model, criterion, optimizer, train_dataloader)
    ##round3
    avg_loss = _run_train_autoencoder(model, criterion, optimizer, train_dataloader)
    ##round4
    avg_loss = _run_train_autoencoder(model, criterion, optimizer, train_dataloader)
    
    torch.save({'state_dict': autoencoder.state_dict()}, 
                "{}/stage_1_autoencoder_loss_{:.4f}.pt".format(args.save_path, avg_loss))
    
    print("\t [stage 1] trained autoencoder, avg reconstruct loss:{:.4f}".format(avg_loss))
    print('\t----------------------------------------------------------')
    
    
    
    ##########################################################################
    #########             [Stage 2] train discriminator              #########    
    
    ##round1
    avg_loss = _run_train_discriminator(model, criterion, optimizer, train_dataloader)
    ##round2
    avg_loss = _run_train_discriminator(model, criterion, optimizer, train_dataloader)
    ##round3
    avg_loss = _run_train_discriminator(model, criterion, optimizer, train_dataloader)
    
    torch.save({'state_dict': discriminator.state_dict()}, 
                "{}/stage_2_discriminat_loss_{:.4f}.pt".format(args.save_path, avg_loss))
    
    print("\t [stage 2] trained discriminator, avg mse loss:{:.4f}".format(avg_loss))
    print('\t----------------------------------------------------------')
    
    
    ##########################################################################
    ########              [Stage 3] adverserial training             #########
    print("\t [stage 3] adversarial training ")
    
    for epoch in range(args.epoch):
        print(f' Epoch {epoch}')
        
        avg_auto_loss, avg_disc_loss = _run_train_adversarial(model, criterion, optimizer, train_dataloader)
        print("\t [Info] Train, avg autoenc loss:{:.4f}, avg discri loss:{:.4f}".format(avg_auto_loss,
                                                                                         avg_disc_loss))
        
        avg_auto_loss, avg_disc_loss = _run_eval(model, criterion, valid_dataloader)
        print("\t [Info] Valid, avg autoenc recons loss:{:.4f},avg discri loss:{:.4f}".format(avg_auto_loss,
                                                                                                   avg_disc_loss))

        if True or loss < best_loss:
            torch.save({'state_dict': autoencoder.state_dict()}, 
                        "{}/epoch_{}_autoencoder_loss_{:.4f}.pt".format(args.save_path, 
                                                                        epoch, avg_auto_loss))
            torch.save({'state_dict': discriminator.state_dict()}, 
                        "{}/epoch_{}_discriminat_loss_{:.4f}.pt".format(args.save_path,
                                                                          epoch, avg_disc_loss))
            print(f'\t [Info] save weights at {args.save_path}/epoch_{epoch}_...')

        print('\t----------------------------------------------------------')
        
    
    
