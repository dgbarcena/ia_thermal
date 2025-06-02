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

n_train = 1500
n_validation = 300
n_test = 50
n_cases = n_train+n_test+n_validation  


nodes_side = 13
time_sim = 1000
seq_len = time_sim+1
dt = 1
T_init = 298.0

n_data = n_cases*seq_len
train_data = n_train*seq_len
validation_data = n_validation*seq_len
test_data = n_test*seq_len

input = np.empty((n_data, 10))
output = np.empty((n_data, nodes_side*nodes_side))

np.random.seed(0)

T_interfaces_random = np.random.uniform(280, 310, (n_cases, 4))
Q_random = np.random.uniform(0.5, 1.0, (n_cases, 4))
T_env_random = np.random.uniform(280, 310, n_cases)
time_column = np.arange(seq_len).reshape(-1, 1)  # Forma (seq_len, 1)

time_start = time.time()

print(F"Generating {n_cases} cases of {seq_len} steps each = {n_data} data")

for i in range(n_cases):
    
    # Print iteration number
    if i%100 == 0:
        print("Generating case number: ", i)
        
    # Generate the data
    T, time_solv, _, _ = PCB_case_2(solver = solver, display=False, time = time_sim, dt = dt, T_init = T_init, Q_heaters = Q_random[i], T_interfaces = T_interfaces_random[i], Tenv = T_env_random[i]) # heaters in default position
    
    # Append the data to the list
    output[i*seq_len:(i+1)*seq_len, :] = T # output is the temperature at each time step
    input1 = np.concatenate((T_interfaces_random[i],Q_random[i],[T_env_random[i]]),axis=0) # input1 is the input data for the case
    input1 =  np.tile(input1, (seq_len, 1)) # input1 is the input data for the case
    input1 = np.hstack((input1, time_column))  # Combina input1 con la nueva columna
    input[i*seq_len:(i+1)*seq_len, :] = input1 # input is the temperature at each time step
    
time_end = time.time()
time_generation_data = time_end-time_start
print("Time to generate the data: ", time_generation_data)

# transform the lists into numpy arrays
input = np.array(input)
output = np.array(output)

input = torch.tensor(input,dtype=torch.float32)
output = torch.tensor(output,dtype=torch.float32)

T_interfaces_random = input[:, :4]
Q_random = input[:, 4:8]
T_env_random = input[:, 8]
time_column = input[:, 9]

Q_heaters = Q_random.clone().detach().float()
T_interfaces = T_interfaces_random.clone().detach().float()
T_env = T_env_random.clone().detach().float()
sequence_length = time_column.clone().detach().float()

# calculate averages and standard deviations
T_interfaces_mean = T_interfaces.mean() # careful because calculated with lots of zeros
T_interfaces_std = T_interfaces.std()
Q_heaters_mean = Q_heaters.mean() # careful because calculated with lots of zeros
Q_heaters_std = Q_heaters.std()
T_env_mean = T_env.mean()
T_env_std = T_env.std()
output_mean = output.mean() 
output_std = output.std()
seq_len_mean = sequence_length.mean()
seq_len_std = sequence_length.std()

dataset = PCBDataset_mlp(T_interfaces, Q_heaters, T_env, sequence_length, output, T_interfaces_mean, T_interfaces_std, Q_heaters_mean, Q_heaters_std, T_env_mean, T_env_std, seq_len_mean, seq_len_std, output_mean, output_std)
dataset_train = dataset[:train_data]
dataset_test = dataset[train_data:train_data+test_data]
dataset_val = dataset[train_data+test_data:]


# path directorie for saving datasets
path = os.path.join(base_path,'datasets')
if not os.path.exists(path):
    os.makedirs(path)
    
# Save the datasets
torch.save(dataset, os.path.join(path, 'PCB_mlp_transient_dataset.pth'))
torch.save(dataset_train, os.path.join(path, 'PCB_mlp_transient_dataset_train.pth'))
torch.save(dataset_test, os.path.join(path, 'PCB_mlp_transient_dataset_test.pth'))
torch.save(dataset_val, os.path.join(path, 'PCB_mlp_transient_dataset_val.pth'))