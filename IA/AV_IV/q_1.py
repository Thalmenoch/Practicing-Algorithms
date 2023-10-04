class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        if len(inputs) == (len(self.weights)):
            raise ValueError("O número de entradas deve ser igual ao número de pesos.")

        # Calcula a soma ponderada das entradas e pesos, incluindo o peso de bias (w0)
        weighted_sum = self.weights[0]  # Começa com o peso de bias (w0)
        for i in range(1, len(inputs)):
            weighted_sum += self.weights[i] * inputs[i-1]

        # Aplica a função de ativação de limiar
        output = 1 if weighted_sum > self.threshold else 0
        return output

if __name__ == '__main__':
    x1 = 10
    x2 = -20
    x3 = -8
    x4 = 2

    # Pesos, incluindo o peso de bias (w0)
    weights = [-1, 0.02, -0.2, 0.03, -0.09]

    threshold = 0

    neuron = McCullochPittsNeuron(weights, threshold)

    inputs = [x1, x2, x3, x4]
    output = neuron.activate(inputs)

    if output == 1:
        print('Classe A')
    else:
        if output == 0:
            print('Classe B')
