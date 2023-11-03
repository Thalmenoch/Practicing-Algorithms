import numpy as np
from MLP import MLP

if __name__ == '__main__':
    # Dados de entrada e saída para a porta NAND
    entradas = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

    # Saídas desejadas para a porta NAND
    saidas_desejadas = np.array([[1],
                                [1],
                                [1],
                                [0]])

    # Inicialização aleatória dos pesos
    np.random.seed(0)
    pesos_camada_entrada = 2 * np.random.random((2, 4)) - 1
    pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

    # Taxa de aprendizado
    taxa_aprendizado = 0.1

    # Número de épocas
    num_epocas = 1000

    # Treinamento
    mlp = MLP(entradas, saidas_desejadas, pesos_camada_entrada, pesos_camada_saida, taxa_aprendizado, num_epocas)
    pesos_camada_entrada, pesos_camada_saida = mlp.treinar_mlp()

    previsoes = mlp.sigmoid(np.dot(mlp.sigmoid(np.dot(entradas, pesos_camada_entrada)), pesos_camada_saida))

    print("Saídas previstas para a porta NAND:")
    for i, previsao in enumerate(previsoes):
        print(f"Entrada: {entradas[i]}, Saída Desejada: {saidas_desejadas[i][0]}, Saída Prevista: {previsao[0]:.2f}")