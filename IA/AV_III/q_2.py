from q_1 import best_k
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    
    x = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ]

    y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  # Resultados da porta XOR

    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.1, random_state=42)

    melhor_k, melhor_acuracia = best_k(x_treino, x_teste, y_treino, y_teste)

    print(f"Melhor valor de k: {melhor_k}")
    print(f"Acurácia do KNN com k={melhor_k}: {melhor_acuracia}")