# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(x, y, output):
#    for binary cross entropy loss
    return (y - output)*x



# Reading the csv file into a pandas DataFrame
data = pd.read_csv('deep learning/datasets/student_data.csv')

one_hot_data = pd.get_dummies(data, columns=['rank'], prefix='rank', drop_first=True)

processed_data = one_hot_data[:]

processed_data  ['gre'] = processed_data['gre'] / 800
processed_data  ['gpa'] = processed_data['gpa'] / 4

sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))

features = train_data.drop('admit', axis=1).astype(float)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1).astype(float)
targets_test = test_data['admit']


# Neural Network hyperparameters
epochs = 1000
learnrate = 0.0001

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output


            # error = error_formula(y, output)

            # The error term
            #   Notice we calulate f'(h) here instead of defining a separate
            #   sigmoidmoid_prime function. This just makes it faster because we
            #   can re-use the result of the sigmoidmoid function stored in
            #   the output variable
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate*del_w #/ n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            features = features.astype(float)
            out = sigmoid(np.dot(features.values, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)

# Calculate accuracy on test data
features_test = features_test.astype(float)
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))