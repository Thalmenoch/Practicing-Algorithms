import numpy as np

def organizar_dados(valor_inicial,valor_final,quantidade):
    # 500 dados igualmente espaçados de 0 a 20
    dados = np.linspace(valor_inicial, valor_final, quantidade)

    
    return dados

    
def formar_dataset(linha,coluna):

    # 500 dados igualmente espaçados de 0 a 20
    dados = organizar_dados(0,20,500)

    # vamos ver como ficaram os dados na forma de array
    # print(f"Dados: \n{dados}")

    # vamos organizar em forma de dataset
    dados_organizados = dados.reshape(linha,coluna)

    # vamos ver como ficou os dados organizados
    return dados_organizados

if __name__ == '__main__':
    # vamos organizar em forma de dataset de 100 x 5
    formar_dataset(100,5)