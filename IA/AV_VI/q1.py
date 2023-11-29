import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função de ativação
def relu(x):
    return np.maximum(0, x)

# Função para treinar a ELM
def train_elm(X_train, y_train, num_neurons=5):
    # Passo 1: Inicialização aleatória dos pesos da camada de entrada
    input_weights = np.random.randn(X_train.shape[1], num_neurons)

    # Passo 2: Calcular a saída da camada oculta
    hidden_output = relu(X_train.dot(input_weights))

    # Passo 3: Calcular os pesos da camada de saída usando a pseudo-inversa
    output_weights = np.linalg.pinv(hidden_output).dot(y_train)

    return input_weights, output_weights

# Função para fazer previsões usando a ELM treinada
def predict_elm(X, input_weights, output_weights):
    hidden_output = relu(X.dot(input_weights))
    predictions = hidden_output.dot(output_weights)
    return np.round(relu(predictions))

if __name__ == "__main__":
    # Definir entradas e saídas
    X = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
    y = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])

    # Dividir dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar a ELM
    input_weights, output_weights = train_elm(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = predict_elm(X_test, input_weights, output_weights)

    # Avaliar a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')
