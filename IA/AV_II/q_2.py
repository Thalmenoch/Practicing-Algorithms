from q_1 import formar_dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

matriz = formar_dataset(100,5)

# Dividir os dados em treinamento (80%) e teste (20%) como mostrado no parâmetro test_size=0.2 (20%)
x_treino, x_teste = train_test_split(matriz, test_size=0.2, random_state=42, shuffle=True) #shuffle garante se a divisão vai ser aleatória

#aplicação da técnica K-fold com k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_indice, val_indice in kf.split(x_treino):
    X_fold_treino, X__fold_val = x_treino[train_indice], x_treino[val_indice]

# print(x_teste)