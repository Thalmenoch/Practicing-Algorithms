import random

class Perceptron:
    def __init__(self, num_inputs):
        # Inicializa os pesos com valores aleatórios entre -1 e 1
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        # Define o limite (threshold) para ativação
        self.threshold = 0

    def activate(self, inputs):
        # Calcula a soma ponderada das entradas e pesos
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        
        # Aplica a função degrau
        output = 1 if weighted_sum >= self.threshold else 0
        return output

# Define a função NAND
if __name__ == '__main__':
    perceptron = Perceptron(2)

    perceptron.weights = [-1, -1]
    perceptron.threshold = -1

    print("NAND(0, 0) =", perceptron.activate([0, 0]))
    print("NAND(0, 1) =", perceptron.activate([0, 1]))
    print("NAND(1, 0) =", perceptron.activate([1, 0]))
    print("NAND(1, 1) =", perceptron.activate([1, 1]))
