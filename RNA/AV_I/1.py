import numpy as np

#classe do modelo de Neurônio de McCulloch-Pitts
class M_P:
    def __init__(self, pesos, limite):
        self.pesos = pesos
        self.limite = limite

    # Cálculo das saídas
    def ativando(self, entrada):
        if len(entrada) != len(self.pesos): # como entrada e o peso são definidos no código, não tem risco dessa exceção ser levantada! 
            raise ValueError("Número de entradas deve ser igual ao número de pesos.")

        # Calcula a soma dos pesos das entradas
        peso_soma = sum(w * x for w, x in zip(self.pesos, entrada))

        # Aplica o limite de ativação da função
        saida = 1 if peso_soma >= self.limite else 0
        return saida

if __name__ == '__main__':
    # Define os pesos e o limite para o portão AND
    and_pesos = np.array([1, 1])
    and_limite = 2

    # Cria uma instância do neurônio de M.P para o portão AND
    and_neuronio = M_P(and_pesos, and_limite)

    # Define combinações de entradas para o portão AND 
    entrada_combinacoes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Testa o neurônio para cada entrada
    for entrada in entrada_combinacoes:
        resultado = and_neuronio.ativando(entrada)
        print(f"Entrada: {entrada}, Saída: {resultado}")