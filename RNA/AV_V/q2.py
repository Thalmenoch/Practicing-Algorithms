import numpy as np

def sigmoid_output(x):
    return 1 / (1 + np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    entradas = np.array([[1, 1],
                        [0, 1],
                        [1, 0],
                        [0, 0]])

    saidas = np.array([1, 0, 0, 0])  

    num_neuronios_oculta = 10

    pesos_oculta = np.random.randn(entradas.shape[1], num_neuronios_oculta)

    ativacoes_oculta = np.dot(entradas, pesos_oculta)


    ativacoes_oculta = sigmoid(ativacoes_oculta)

    pesos_saida = np.linalg.lstsq(ativacoes_oculta, saidas, rcond=None)[0]

    saidas_pred = np.dot(ativacoes_oculta, pesos_saida)


    saidas_pred = sigmoid_output(saidas_pred)

    print("Saídas previstas:")
    print(saidas_pred)
