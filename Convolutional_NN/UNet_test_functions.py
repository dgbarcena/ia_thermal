import torch
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import random

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
from plot_functions import *


#######################################################################################################################
############################################### UNET_test_test() ######################################################
#######################################################################################################################

def UNET_test_test(model_name:str,finetuning_model_name:str=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    dataset_path = os.path.join(dir_path,'Datasets')

    if finetuning_model_name == None:
        folder_path = os.path.join(models_path,'UNET',model_name)
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',model_name,finetuning_model_name)


    # Importación de los dataset de train y test.
    if finetuning_model_name == None:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test.pth'),weights_only=False)
    else:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test_finetuning.pth'),weights_only=False)
    #dataset_test.to_cuda()

    n_test = dataset_test.inputs.size(0)
    test_sampler = SubsetRandomSampler(list(range(n_test)))

    test_loader = DataLoader(dataset_test, batch_size=1, sampler=test_sampler)

    # Creación de los modelos.
    unet = UNet()
    try:
        unet.load_state_dict(torch.load(os.path.join(folder_path,'UNet.pth'),weights_only=True))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))
    unet = unet.to(device)

    # Definimos la pérdida de reconstrucción del autoencoder.
    criterionReconstruction = nn.MSELoss()
    
    # Bucle de test.

    unet.eval()

    with torch.no_grad():
        # Bucle Test.
        unet.eval()
        total_loss = 0
        total_max_loss = 0
        max_loss_list = []
        rmse_loss_list = []
        with torch.no_grad():
            for i in range(n_test):
                input,target = dataset_test.__getitem__(i)
                target = target.view(-1,1,13,13)
                input = input.view(-1,3,13,13)

                x = unet(input)

                x = x.view(-1,13,13)
                target = target.view(-1,13,13)
                x = dataset_test.denormalize_output(x)
                target = dataset_test.denormalize_output(target)
                g_loss = criterionReconstruction(x,target) 
                max_loss = torch.max(torch.abs(x-target))
                rmse_loss_list.append(g_loss.item())
                max_loss_list.append(max_loss.item())
                
                loss = g_loss

                total_loss += loss.item() 
                total_max_loss += max_loss

    # Calcular la pérdida promedio
    RMSE_test = np.sqrt(total_loss / len(test_loader)) 
    max_test = total_max_loss / len(test_loader)
    print(f"Average RMSE test loss for UNet {folder_path} = {RMSE_test:.8f}") 
    print(f"Average max test loss for UNet {folder_path} = {max_test:.8f}") 

    dict = {'rmse_loss': rmse_loss_list, 'max_loss': max_loss_list}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder_path,'Results','test_results.csv'), index=False)

    dict = {'rmse_loss':[RMSE_test],'max_test':[max_test.item()]}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder_path,'Results','test_results_mean.csv'), index=False)


    return RMSE_test,max_test



#######################################################################################################################
############################################## UNET_test_train() ######################################################
#######################################################################################################################

def UNET_test_train(model_name:str,n_train:int,finetuning_model_name:str=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    dataset_path = os.path.join(dir_path,'Datasets')
    if finetuning_model_name == None:
        folder_path = os.path.join(models_path,'UNET',model_name)
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',model_name,finetuning_model_name)


    # Importación de los dataset de train y test.
    if finetuning_model_name == None:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test.pth'),weights_only=False)
    else:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test_finetuning.pth'),weights_only=False)
    #dataset_test.to_cuda()

    n_test = n_train
    test_sampler = SubsetRandomSampler(list(range(n_test)))

    test_loader = DataLoader(dataset_test, batch_size=1, sampler=test_sampler)

    # Creación de los modelos.
    unet = UNet()
    try:
        unet.load_state_dict(torch.load(os.path.join(folder_path,'UNet.pth'),weights_only=True))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))
    unet = unet.to(device)

    # Definimos la pérdida de reconstrucción del autoencoder.
    criterionReconstruction = nn.MSELoss()
    
    # Bucle de test.

    unet.eval()

    with torch.no_grad():
        # Bucle Test.
        unet.eval()
        total_loss = 0
        total_max_loss = 0
        max_loss_list = []
        rmse_loss_list = []
        with torch.no_grad():
            for i in range(n_test):
                input,target = dataset_test.__getitem__(i)
                target = target.view(-1,1,13,13)
                input = input.view(-1,3,13,13)

                x = unet(input)

                x = x.view(-1,13,13)
                target = target.view(-1,13,13)
                x = dataset_test.denormalize_output(x)
                target = dataset_test.denormalize_output(target)
                g_loss = criterionReconstruction(x,target) 
                max_loss = torch.max(torch.abs(x-target))
                rmse_loss_list.append(g_loss.item())
                max_loss_list.append(max_loss.item())
                
                loss = g_loss

                total_loss += loss.item() 
                total_max_loss += max_loss 

    # Calcular la pérdida promedio
    RMSE_test = np.sqrt(total_loss / len(test_loader)) 
    max_test = total_max_loss / len(test_loader)
    print(f"Average RMSE train loss for UNet {folder_path} = {RMSE_test:.8f}") 
    print(f"Average max train loss for UNet {folder_path} = {max_test:.8f}") 

    dict = {'rmse_loss': rmse_loss_list, 'max_loss': max_loss_list}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder_path,'Results','train_results.csv'), index=False)

    dict = {'rmse_loss':[RMSE_test],'max_test':[max_test.item()]}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(folder_path,'Results','train_results_mean.csv'), index=False)

    return RMSE_test,max_test



#######################################################################################################################
########################################### UNET_plot_sample_test() ###################################################
#######################################################################################################################

def UNET_plot_sample_test(model_name:str,finetuning_model_name:str=None,sample_number = 'random'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    dataset_path = os.path.join(dir_path,'Datasets')

    if finetuning_model_name == None:
        folder_path = os.path.join(models_path,'UNET',model_name)
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',model_name,finetuning_model_name)


    # Importación de los dataset de train y test.
    if finetuning_model_name == None:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test.pth'),weights_only=False)
    else:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test_finetuning.pth'),weights_only=False)
    #dataset_test.to_cuda()

    n_test = dataset_test.inputs.size(0)

    # Creación de los modelos.
    unet = UNet()
    try:
        unet.load_state_dict(torch.load(os.path.join(folder_path,'UNet.pth'),weights_only=True))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))
    unet = unet.to(device)

    # Bucle de test.

    unet.eval()

    # Plot caso aleatorio.
    if sample_number == 'random':
        rn = random.choice(list(range(n_test)))
    else:
        rn = sample_number

    with torch.no_grad():
        # Bucle Test.
        unet.eval()
        
        input,target = dataset_test.__getitem__(rn)
        target = target.view(-1,1,13,13)
        input = input.view(-1,3,13,13)
        x = unet(input)
        x = x.view(-1,13,13)
        target = target.view(-1,13,13)
        x = dataset_test.denormalize_output(x)
        target = dataset_test.denormalize_output(target)
        print(torch.max(torch.abs(x-target)))
        plot_sample(x,target)



#######################################################################################################################
########################################### UNET_plot_sample_test() ###################################################
#######################################################################################################################

def UNET_plot_sample_train(model_name:str,n_train:int,finetuning_model_name:str=None,sample_number='random'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    dataset_path = os.path.join(dir_path,'Datasets')
    if finetuning_model_name == None:
        folder_path = os.path.join(models_path,'UNET',model_name)
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',model_name,finetuning_model_name)


    # Importación de los dataset de train y test.
    if finetuning_model_name == None:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test.pth'),weights_only=False)
    else:
        dataset_test:PCBDataset = torch.load(os.path.join(dataset_path,'PCB_dataset_test_finetuning.pth'),weights_only=False)
    #dataset_test.to_cuda()

    n_test = n_train

    # Creación de los modelos.
    unet = UNet()
    try:
        unet.load_state_dict(torch.load(os.path.join(folder_path,'UNet.pth'),weights_only=True))
    except:
        raise(Exception("Exception occurred: there are no saved models with the desired properties"))
    unet = unet.to(device)

    # Bucle de test.

    unet.eval()

    # Plot caso aleatorio.
    if sample_number == 'random':
        rn = random.choice(list(range(n_test)))
    else:
        rn = sample_number
        
    with torch.no_grad():
        # Bucle Test.
        unet.eval()
        
        input,target = dataset_test.__getitem__(rn)
        target = target.view(-1,1,13,13)
        input = input.view(-1,3,13,13)
        x = unet(input)
        x = x.view(-1,13,13)
        target = target.view(-1,13,13)
        x = dataset_test.denormalize_output(x)
        target = dataset_test.denormalize_output(target)
        print(torch.max(torch.abs(x-target)))
        plot_sample(x,target)
