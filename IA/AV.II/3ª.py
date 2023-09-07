import numpy as np

def organizar_dados(valor_inicial,valor_final,quantidade):
    # 500 dados igualmente espaçados de 0 a 20
    dados = np.linspace(valor_inicial, valor_final, quantidade)

    
    return dados

    
def formar_dataset(linha,coluna):

    # 500 dados igualmente espaçados de 0 a 20
    dados = organizar_dados(0,20,500)

    # vamos organizar em forma de dataset
    dados_organizados = dados.reshape(linha,coluna)

    # vamos ver como ficou os dados organizados
    return dados_organizados

def zscore():

    dataset = formar_dataset(100,5)

    # Calcular a média e o desvio padrão ao longo das colunas (axis=0)
    media = np.mean(dataset, axis=0)
    desvio_padrao = np.std(dataset, axis=0)

    # Calcular o Z-score
    zscore = (dataset - media) / desvio_padrao

    print(zscore)

zscore()