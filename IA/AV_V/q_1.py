import numpy as np
from MLP import MLP

if __name__ == '__main__':
    # Dados de entrada e saída
    entradas = np.array([[1, 1, 1],
                        [0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])

    saidas_desejadas = np.array([[1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0]])

    # Inicialização aleatória dos pesos
    np.random.seed(0)
    pesos_camada_entrada = 2 * np.random.random((3, 4)) - 1
    pesos_camada_saida = 2 * np.random.random((4, 1)) - 1

    taxa_aprendizado = 0.1

    num_epocas = 1000

    pesos_camada_entrada, pesos_camada_saida = MLP.treinar_mlp(entradas, saidas_desejadas, pesos_camada_entrada, pesos_camada_saida, taxa_aprendizado, num_epocas)

    previsoes = MLP.sigmoid(np.dot(MLP.sigmoid(np.dot(entradas, pesos_camada_entrada)), pesos_camada_saida))

    print("Saídas previstas:")
    for i, previsao in enumerate(previsoes):
        print(f"Entrada: {entradas[i]}, Saída Desejada: {saidas_desejadas[i][0]}, Saída Prevista: {previsao[0]:.2f}")
