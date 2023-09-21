import numpy as np

# Propagação direta (forward pass)
def forward_pass(X):
    # Camada oculta
    Z1 = np.dot(X, W1)
    A1 = np.tanh(Z1)
    
    # Camada de saída
    Z2 = np.dot(A1, W2)
    A2 = np.tanh(Z2)
    
    return A2

if __name__ == '__main__':
    # Definição dos dados de entrada (porta NOR com 3 entradas)
    X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    np.random.seed(146)
    input_size = 3
    hidden_size = 2  
    output_size = 1

    W1 = np.random.normal(scale=1, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=1, size=(hidden_size, output_size))

    output = forward_pass(X)

    rounded_output = np.round(output)

    print("Saídas arredondadas:")
    print(rounded_output)