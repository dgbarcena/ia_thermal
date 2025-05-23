import numpy as np
import os
import sys
import time
import torch

base_path = os.path.dirname(__file__)

# Añadir 'scripts'
script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(os.path.abspath(script_path))

# Añadir 'Convolutional_NN'
cnn_path = os.path.abspath(os.path.join(base_path, '..', 'Convolutional_NN'))
if cnn_path not in sys.path:
    sys.path.append(cnn_path)

from PCB_solver_tr import PCB_case_2
from Dataset_Class import PCBDataset

solver = 'steady' # steady or transient

n_train = 2048
n_validation = 20000
n_test = 10000
n_data = n_train+n_test+n_validation  

nodes_side = 13
time_sim = 100
dt = 1
T_init = 298.0

input = []
output = []

np.random.seed(0)

Q_random = np.random.uniform(0.5, 1.0, (n_data, 4))
T_interfaces_random = np.random.uniform(280, 310, (n_data, 4))
T_env_random = np.random.uniform(280, 310, n_data)

time_start = time.time()

for i in range(n_data):
    
    # Print iteration number
    if i%200 == 0:
        print("Generating element number: ",i)
        
    # Generate the data
    T, _, _, _ = PCB_case_2(solver = solver, display=False, time = time_sim, dt = dt, T_init = T_init, Q_heaters = Q_random[i], T_interfaces = T_interfaces_random[i], Tenv = T_env_random[i]) # heaters in default position
    # T = T.reshape(T.shape[0], nodes_side,nodes_side) # reshaping the data grid-shape
    
    # Append the data to the list
    output.append(T)
    input1 = []
    # print(T_interfaces_random[i],Q_random[i],T_env_random[i]) # DEBUGGING
    input1 = np.concatenate((T_interfaces_random[i],Q_random[i],[T_env_random[i]]),axis=0)
    # print(input1) # DEBUGGING
    input.append(input1)
    
time_end = time.time()
time_generation_data = time_end-time_start
print("Time to generate the data: ",time_generation_data)

# transform the lists into numpy arrays
input = np.array(input)
output = np.array(output)

if solver == 'transient':
    output = output.reshape(output.shape[0], output.shape[1], nodes_side,nodes_side) # reshaping the data grid-shape
elif solver == 'steady':
    output = output.reshape(output.shape[0], nodes_side,nodes_side) # reshaping the data grid-shape
else:
    raise ValueError("Solver must be 'transient' or 'steady'")

input = torch.tensor(input,dtype=torch.float32)
output = torch.tensor(output,dtype=torch.float32)

T_interfaces = np.zeros((n_data, nodes_side,nodes_side))
Q_heaters = np.zeros((n_data, nodes_side,nodes_side))
T_env = np.zeros((n_data, nodes_side,nodes_side))

for i in range(n_data):
    Q_heaters[i,6,3], Q_heaters[i,3,6],Q_heaters[i,9,3], Q_heaters[i,9,9] = Q_random[i]
    T_interfaces[i,0,0], T_interfaces[i,0,nodes_side-1], T_interfaces[i,nodes_side-1,nodes_side-1], T_interfaces[i,nodes_side-1,0] = T_interfaces_random[i]
    T_env[i,:,:] = T_env_random[i]
    
Q_heaters = torch.tensor(Q_heaters,dtype=torch.float32)
T_env = torch.tensor(T_env,dtype=torch.float32)
T_interfaces = torch.tensor(T_interfaces,dtype=torch.float32)

# calculate averages and standard deviations
T_interfaces_mean = T_interfaces.mean() # careful because calculated with lots of zeros
T_interfaces_std = T_interfaces.std()
Q_heaters_mean = Q_heaters.mean() # careful because calculated with lots of zeros
Q_heaters_std = Q_heaters.std()
T_env_mean = T_env.mean()
T_env_std = T_env.std()
output_mean = output.mean() 
output_std = output.std()


dataset_test = PCBDataset(T_interfaces[:n_test,:,:],Q_heaters[:n_test,:,:],T_env[:n_test,:,:],output[:n_test,:,:],
                 T_interfaces_mean,T_interfaces_std,Q_heaters_mean,
                 Q_heaters_std,T_env_mean,T_env_std,output_mean,
                 output_std)

dataset_train = PCBDataset(T_interfaces[n_test:-n_validation,:,:],Q_heaters[n_test:-n_validation,:,:],T_env[n_test:-n_validation,:,:],output[n_test:-n_validation,:,:],
                 T_interfaces_mean,T_interfaces_std,Q_heaters_mean,
                 Q_heaters_std,T_env_mean,T_env_std,output_mean,
                 output_std)

dataset_val = PCBDataset(T_interfaces[-n_validation:,:,:],Q_heaters[-n_validation:,:,:],T_env[-n_validation:,:,:],output[-n_validation:,:,:],
                 T_interfaces_mean,T_interfaces_std,Q_heaters_mean,
                 Q_heaters_std,T_env_mean,T_env_std,output_mean,
                 output_std)

dataset = PCBDataset(T_interfaces,Q_heaters,T_env,output,
                 T_interfaces_mean,T_interfaces_std,Q_heaters_mean,
                 Q_heaters_std,T_env_mean,T_env_std,output_mean,
                 output_std)

# path directorie for saving datasets
path = os.path.join(base_path,'datasets')
if not os.path.exists(path):
    os.makedirs(path)
    
# torch.save(dataset_train, os.path.join(path, 'PCB_transient_dataset_train.pth'))
# torch.save(dataset_test, os.path.join(path, 'PCB_transient_dataset_test.pth'))
# torch.save(dataset_val, os.path.join(path, 'PCB_transient_dataset_val.pth'))
# torch.save(dataset, os.path.join(path, 'PCB_transient_dataset.pth'))

# torch.save(dataset_train, os.path.join(path, 'PCB_transient_dataset_train_reducedrange.pth'))
# torch.save(dataset_test, os.path.join(path, 'PCB_transient_dataset_test_reducedrange.pth'))
# torch.save(dataset_val, os.path.join(path, 'PCB_transient_dataset_val_reducedrange.pth'))
# torch.save(dataset, os.path.join(path, 'PCB_transient_dataset_reducedrange.pth'))

torch.save(dataset_train, os.path.join(path, 'PCB_steady_dataset_train.pth'))
torch.save(dataset_test, os.path.join(path, 'PCB_steady_dataset_test.pth'))
torch.save(dataset_val, os.path.join(path, 'PCB_steady_dataset_val.pth'))
torch.save(dataset, os.path.join(path, 'PCB_steady_dataset.pth'))