import torch
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

#Pytorch dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# PyTorch model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Dataset_Class import PCBDataset
from NN_Models import *
from Physics_Loss import *

import pandas as pd


def UNET_finetuning_train(pretrained_model:str,epochs:int,n_train:int=1000,batch_size:int=5,lr0:float=0.0001,overwrite:bool = False,
                          only_last_layers:bool = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    dataset_path = os.path.join(dir_path,'Datasets')
    model_path = os.path.join(models_path,'UNET',pretrained_model)
    model_path_fn = os.path.join(models_path,'UNET_finetuning',pretrained_model)
    if only_last_layers == False:
        folder_path = os.path.join(models_path,'UNET_finetuning',pretrained_model,'case_e_{}_nt_{}_bs_{}'.format(epochs,n_train,batch_size))
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',pretrained_model,'case_e_{}_nt_{}_bs_{}_last_layers'.format(epochs,n_train,batch_size))

    # creación del directorio en el que guardar el caso.
    if os.path.exists(folder_path):
        if overwrite == False:
            raise Exception("Case already run. If you want to overwrite it change 'overwrite=True'")
        else:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
            os.mkdir(os.path.join(folder_path,'Results'))
    else:
        if not os.path.exists(model_path_fn):
            os.mkdir(model_path_fn)
        os.mkdir(folder_path)
        os.mkdir(os.path.join(folder_path,'Results'))

    # Copiamos la UNet preentrenda en la nueva carpeta.
    try:
        shutil.copy(os.path.join(model_path,'UNet.pth'),os.path.join(folder_path,'UNet.pth'))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))

    # Importación de los dataset de train y test.
    dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test_finetuning.pth'),weights_only=False)
    dataset_train:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_train_finetuning.pth'),weights_only=False)

    #dataset_train.to_cuda()
    #dataset_test.to_cuda()

    n_test = dataset_test.inputs.size(0)
    train_sampler = SubsetRandomSampler(list(range(n_train)))
    test_sampler = SubsetRandomSampler(list(range(n_test)))

    train_loader = DataLoader(dataset_train, batch_size=5, sampler=train_sampler)
    test_loader = DataLoader(dataset_test, batch_size=5, sampler=test_sampler)

    # Creación de los modelos.
    unet = UNet()
    try:
        unet.load_state_dict(torch.load(os.path.join(folder_path,'UNet.pth'),weights_only=True))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))
    unet = unet.to(device)

    if only_last_layers == True: 
        # bloqueamos la parte de la red que no queremos entrenar de nuevo. 
        for param in unet.parameters():
            param.requires_grad = False
        for param in unet.outc.parameters():
            param.requires_grad = True
        for param in unet.up3.parameters():
            param.requires_grad = True

        # Definimos el optimizador y la pérdida de reconstrucción del autoencoder.
        optimizer = optim.Adam(list(unet.up3.parameters())+list(unet.outc.parameters()), lr=lr0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterionReconstruction = nn.MSELoss().to(device)
    else:
        # Definimos el optimizador y la pérdida de reconstrucción del autoencoder.
        optimizer = optim.Adam(list(unet.parameters()), lr=lr0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterionReconstruction = nn.MSELoss().to(device)

    # Bucle de entrenamiento de la red
    num_train_batches = len(train_loader)
    num_test_batches = len(test_loader)
    last_test_loss = np.inf
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):

        # Bucle Train.
        unet.train() 
        total_loss = 0
        for input,target in train_loader:

            optimizer.zero_grad()

            target = target.view(-1,1,13,13)
            input = input.view(-1,3,13,13)

            x = unet(input)

            x = x.view(-1,13,13)
            target = target.view(-1,13,13)

            g_loss = criterionReconstruction(x,target) 
            
            loss = g_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() 

        avg_train_loss = total_loss/num_train_batches
        train_loss_list.append(avg_train_loss)

        # Bucle Test.
        unet.eval()
        total_loss = 0
        with torch.no_grad():
            for input,target in test_loader:

                target = target.view(-1,1,13,13)
                input = input.view(-1,3,13,13)
                
                x = unet(input)

                x = x.view(-1,13,13)
                target = target.view(-1,13,13)

                g_loss = criterionReconstruction(x,target) 
                
                loss = g_loss

                total_loss += loss.item() 

            avg_test_loss = total_loss/num_test_batches

        scheduler.step(avg_test_loss)
        last_lr = scheduler.optimizer.param_groups[0]['lr']
        test_loss_list.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.8f}, Current LR: {last_lr}")
        
        if avg_test_loss < last_test_loss:
            print("{:.8f} -----> {:.8f}   Saving...".format(last_test_loss,avg_test_loss))
            torch.save(unet.state_dict() , os.path.join(folder_path,'UNet.pth'))
            last_test_loss = avg_test_loss

    dict = {'train_loss': train_loss_list, 'test_loss': test_loss_list}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder_path,'Results','train_loss.csv'), index=False)