import numpy as np

class Adaline:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)  # +1 para o peso do viés (bias)

    def predict(self, inputs):
        # Função de ativação: Linear (identidade)
        net_input = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if net_input >= 0 else 0

    def train(self, training_data, target):
        for _ in range(self.epochs):
            for inputs in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

def main():
    # Defina os dados de treinamento para o problema "NAND"
    training_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    target = np.array([1, 1, 1, 0])  # Saída esperada para o NAND

    # Crie e treine a Adaline
    adaline = Adaline(input_size=2)
    adaline.train(training_data, target)

    # Teste a rede treinada
    test_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    print("Resultado do NAND:")
    for inputs in test_data:
        prediction = adaline.predict(inputs)
        print(f"Entradas: {inputs}, Saída: {prediction}")

if __name__ == "__main__":
    main()