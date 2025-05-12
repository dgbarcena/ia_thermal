import numpy as np
import os
import sys
import time
import torch

from Dataset_Class_mlp import PCBDataset_mlp

base_path = os.path.dirname(__file__)

# Añadir 'scripts'
script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(os.path.abspath(script_path))

# Añadir 'Convolutional_NN'
cnn_path = os.path.abspath(os.path.join(base_path, '..', 'Convolutional_NN'))
if cnn_path not in sys.path:
    sys.path.append(cnn_path)

from PCB_solver_tr import PCB_case_2

solver = 'transient' # steady or transient

n_train = 100000
n_validation = 30000
n_test = 10000
n_data = n_train+n_test+n_validation  

nodes_side = 13
time_sim = 1000
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
print("Time to generate the data: ", time_generation_data)

# transform the lists into numpy arrays
input = np.array(input)
output = np.array(output)

# if solver == 'transient':
#     output = output.reshape(output.shape[0], output.shape[1]) # reshaping the data grid-shape
# elif solver == 'steady':
#     output = output.reshape(output.shape[0], nodes_side,nodes_side) # reshaping the data grid-shape
# else:
#     raise ValueError("Solver must be 'transient' or 'steady'")

input = torch.tensor(input,dtype=torch.float32)
output = torch.tensor(output,dtype=torch.float32)

T_interfaces = np.zeros((n_data, 4))
Q_heaters = np.zeros((n_data, 4))
T_env = np.zeros((n_data))

# for i in range(n_data):
#     Q_heaters[i] = Q_random[i]
#     T_interfaces[i] = T_interfaces_random[i]
#     T_env[i,:,:] = T_env_random[i]

# Q_random_shape = Q_random.shape
# print("Q_random shape: ", Q_random_shape)
    
Q_heaters = torch.tensor(Q_random,dtype=torch.float32)
T_interfaces = torch.tensor(T_interfaces_random,dtype=torch.float32)
T_env = torch.tensor(T_env_random,dtype=torch.float32)

# print("Q_heaters shape: ", Q_heaters.shape)
# print("T_env shape: ", T_env.shape)

# calculate averages and standard deviations
T_interfaces_mean = T_interfaces.mean() # careful because calculated with lots of zeros
T_interfaces_std = T_interfaces.std()
Q_heaters_mean = Q_heaters.mean() # careful because calculated with lots of zeros
# print("Q_heaters_mean: ", Q_heaters_mean)
Q_heaters_std = Q_heaters.std()
T_env_mean = T_env.mean()
T_env_std = T_env.std()
output_mean = output.mean() 
output_std = output.std()

# print("T_interfaces_mean: ", T_interfaces_mean)
# print("T_interfaces_std: ", T_interfaces_std)
# print("Q_heaters_mean: ", Q_heaters_mean)
# print("Q_heaters_std: ", Q_heaters_std)
# print("T_env_mean: ", T_env_mean)
# print("T_env_std: ", T_env_std)
# print("output_mean: ", output_mean)
# print("output_std: ", output_std)

# print("T_interfaces shape fuera: ", T_interfaces.shape)
dataset = PCBDataset_mlp(T_interfaces, Q_heaters, T_env, output, T_interfaces_mean, T_interfaces_std, Q_heaters_mean, Q_heaters_std, T_env_mean, T_env_std, output_mean, output_std)

dataset_train = dataset[:n_train]
dataset_test = dataset[n_train:n_train+n_test]
dataset_val = dataset[n_train+n_test:n_train+n_test+n_validation]


# path directorie for saving datasets
path = os.path.join(base_path,'datasets')
if not os.path.exists(path):
    os.makedirs(path)
    
# Save the datasets
torch.save(dataset, os.path.join(path, 'PCB_mlp_transient_dataset.pth'))
torch.save(dataset_train, os.path.join(path, 'PCB_mlp_transient_dataset_train.pth'))
torch.save(dataset_test, os.path.join(path, 'PCB_mlp_transient_dataset_test.pth'))
torch.save(dataset_val, os.path.join(path, 'PCB_mlp_transient_dataset_val.pth'))