import numpy as np

# Função Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da Função
def sigmoid_derivative(x):
    return x * (1 - x)

def backpropagation(learning_rate, epochs, x, y, synapse_0, synapse_1):
    for epoch in range(epochs):
        # Etapa de feedforward
        layer_0 = x
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # Cálculo do erro
        error = y - layer_2

        # Etapa de retropropagação
        delta_2 = error * sigmoid_derivative(layer_2)
        delta_1 = delta_2.dot(synapse_1.T) * sigmoid_derivative(layer_1)

        # Atualização dos pesos
        synapse_1 += layer_1.T.dot(delta_2) * learning_rate
        synapse_0 += layer_0.T.dot(delta_1) * learning_rate
    
    return layer_2

if __name__ == '__main__':
    x = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]])

    y = np.array([[1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [0]])

    # Inicialização aleatória dos pesos e bias
    np.random.seed(1)

    # Tamanhos das Camadas
    input_layer_size = 3
    hidden_layer_size = 4
    output_layer_size = 1

    # Pesos da camada de entrada para a camada oculta
    synapse_0 = 2 * np.random.random((input_layer_size, hidden_layer_size)) - 1

    # Pesos da camada oculta para a camada de saída
    synapse_1 = 2 * np.random.random((hidden_layer_size, output_layer_size)) - 1

    learning_rate = 0.1
    epochs = 1000

    layer_2 = backpropagation(learning_rate, epochs, x, y, synapse_0, synapse_1)

    # Resultados após o treinamento
    print("Saída após o treinamento:")
    print(layer_2)
