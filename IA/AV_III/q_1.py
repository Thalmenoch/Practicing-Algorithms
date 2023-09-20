from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def best_k(x_treino, x_teste, y_treino, y_teste):
    melhor_k = None
    melhor_acuracia = 0

    # Teste de K
    for k in range(1, len(x_treino) + 1):
        knn = KNeighborsClassifier(n_neighbors=k) # aplicação do knn
        knn.fit(x_treino, y_treino)
        y_predizido = knn.predict(x_teste)
        acuracia  = accuracy_score(y_teste, y_predizido)
        
        if acuracia  > melhor_acuracia:
            melhor_acuracia = acuracia 
            melhor_k = k
        
    return melhor_k, melhor_acuracia

if __name__ == '__main__':
    
    x = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0]
    ]

    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # Resultados da porta NAND

    # Divisão de dados
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.1, random_state=42)

    melhor_k, melhor_acuracia = best_k(x_treino, x_teste, y_treino, y_teste)

    print(f"Melhor valor de k: {melhor_k}")
    print(f"Acurácia do KNN com k={melhor_k}: {melhor_acuracia}")