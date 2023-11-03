import numpy as np

# Defina as entradas e saídas
entradas = np.array([[1, 1, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])

saidas = np.array([1, 1, 1, 1, 1, 1, 1, 0])

# Defina o número de neurônios na camada oculta
num_neuronios_oculta = 10

# Inicialize os pesos aleatórios das conexões entre a camada de entrada e a camada oculta
pesos_oculta = np.random.randn(entradas.shape[1], num_neuronios_oculta)

# Calcule as ativações da camada oculta
ativacoes_oculta = np.dot(entradas, pesos_oculta)

# Aplique uma função de ativação, por exemplo, a função sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

ativacoes_oculta = sigmoid(ativacoes_oculta)

# Resolva o sistema linear para calcular os pesos da camada de saída
pesos_saida = np.linalg.lstsq(ativacoes_oculta, saidas, rcond=None)[0]

# Calcule as saídas da rede ELM
saidas_pred = np.dot(ativacoes_oculta, pesos_saida)

# Aplicar uma função de ativação para mapear as saídas para o intervalo [0, 1] (opcional)
def sigmoid_output(x):
    return 1 / (1 + np.exp(-x))

saidas_pred = sigmoid_output(saidas_pred)

# Agora, você pode usar saidas_pred para fazer previsões
print("Saídas previstas:")
print(saidas_pred)
