import numpy as np

# Função de ativação (função degrau)
def step_function(x):
    if x >= 0:
        return 1
    else:
        return 0

# Definindo os pesos e o viés para a porta OR
weights = np.array([1, 1])  # Pesos positivos
bias = -0.5  # Viés

# Função para calcular a saída do perceptron
def perceptron(input_values, weights, bias):
    net_input = np.dot(input_values, weights) + bias
    output = step_function(net_input)
    return output

# Função para testar a porta OR
def test_or_gate():
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    print("Input 1\tInput 2\tOR Output")

    for inputs in input_data:
        input1, input2 = inputs
        output = perceptron(inputs, weights, bias)
        print(f"{input1}\t{input2}\t{output}")

if __name__ == "__main__":
    test_or_gate()
