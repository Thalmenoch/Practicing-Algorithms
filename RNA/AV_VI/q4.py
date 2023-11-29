import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para gerar dados para a porta XOR
def gerando_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    return X, y

# Função para treinar a RNA RBF
def treinar_rbf(X_train, y_train, n_clusters=2):
    # Passo 1: Clusterização usando K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    centers = kmeans.cluster_centers_

    # Passo 2: Calcular as funções de base radial (RBF)
    rbf_functions = np.exp(-0.1 * np.sum((X_train[:, np.newaxis] - centers) ** 2, axis=2))

    # Passo 3: Treinar a camada de saída usando perceptron multicamadas (MLP)
    rbf_mlp = MLPClassifier(hidden_layer_sizes=(n_clusters,), activation='relu', max_iter=1000, random_state=42)
    rbf_mlp.fit(rbf_functions, y_train)

    return centers, rbf_mlp

# Função para fazer previsões usando a RNA treinada
def predicao_rbf(X, centers, rbf_mlp):
    rbf_functions = np.exp(-0.1 * np.sum((X[:, np.newaxis] - centers) ** 2, axis=2))
    return rbf_mlp.predict(rbf_functions)

if __name__ == "__main__":
        # Gerar dados de treinamento e teste
    X, y = gerando_xor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar a RNA RBF
    centers, rbf_mlp = treinar_rbf(X_train, y_train)

    # Fazer previsões nos dados de teste
    y_pred = predicao_rbf(X_test, centers, rbf_mlp)

    # Avaliar a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

