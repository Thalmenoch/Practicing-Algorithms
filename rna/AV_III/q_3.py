import numpy as np

weights_input_hidden = np.array([[0.2, -0.1, 0.4],
                                  [0.7, -1.2, 1.2]])
bias_hidden = np.array([-1, -1])

weights_hidden_output = np.array([[1.1, 0.1],
                                   [3.1, 1.17]])
bias_output = np.array([-1, -1])

def linear_activation(x):
    return x

def forward_propagation(input_data):
    hidden_input = np.dot(weights_input_hidden, input_data) + bias_hidden
    hidden_output = linear_activation(hidden_input)

    hidden2_input = np.dot(weights_hidden_output, hidden_output) + bias_output
    hidden2_output = linear_activation(hidden2_input)

    return hidden2_output

input1 = np.array([10, 12, -9])
input2 = np.array([-2, 3, 30])

output1 = forward_propagation(input1)
output2 = forward_propagation(input2)

if output1[0] >= output1[1]:
    classe_input1 = 1
else:
    classe_input1 = 2

if output2[0] >= output2[1]:
    classe_input2 = 1
else:
    classe_input2 = 2

print("Classe estimada para o input1:", classe_input1)
print("Classe estimada para o input2:", classe_input2)