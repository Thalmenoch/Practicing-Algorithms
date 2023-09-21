import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Propagação direta (forward pass)
def forward_pass(x):
    # Camada oculta
    Z1 = np.dot(x, W1)
    A1 = sigmoid(Z1)
    
    # Camada de saída
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    
    return A2

if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Pesos gerados aleatoriamente
    np.random.seed(146)
    input_size = 2
    hidden_size = 2
    output_size = 1

    W1 = np.random.normal(scale=1, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=1, size=(hidden_size, output_size))

    output = forward_pass(x)

    # Arredonda as saídas para 0 ou 1
    rounded_output = np.round(output)

    print("Saídas arredondadas:")
    print(rounded_output)
