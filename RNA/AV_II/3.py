import numpy as np

input_size = 3
learning_rate = 0.1
epochs = 100

weights = np.zeros(input_size + 1)

training_data = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 1, 1],
                          [1, 0, 0],
                          [1, 0, 1],
                          [1, 1, 0],
                          [1, 1, 1]])

labels = np.array([0, 1, 1, 1, 1, 1, 1, 1])

for _ in range(epochs):
    for i in range(len(training_data)):
        inputs = np.insert(training_data[i], 0, 1)
        prediction = np.dot(inputs, weights)
        error = labels[i] - (prediction >= 0)
        weights += learning_rate * error * inputs

test_inputs = np.array([[0, 0, 1],
                        [1, 1, 0],
                        [0, 0, 0]])

for inputs in test_inputs:
    inputs_with_bias = np.insert(inputs, 0, 1)
    summation = np.dot(inputs_with_bias, weights)
    prediction = 1 if summation >= 0 else 0
    print(f"Entradas: {inputs}, Saída Prevista: {prediction}")


