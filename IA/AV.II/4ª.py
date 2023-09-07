import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def ordenacao(dados1, dados2):
    melhor_ordem = None
    melhor_r2 = -np.inf
    melhor_eqm = np.inf

    for ordem in range(1, 10):  
        coeficientes = np.polyfit(dados1, dados2, ordem)
        y_previsto = np.polyval(coeficientes, dados1)
    
        r2 = r2_score(dados2, y_previsto)
        eqm = mean_squared_error(dados2, y_previsto)
        
        if r2 > melhor_r2:
            melhor_r2 = r2
            melhor_ordem = ordem
            melhor_eqm = eqm
        
        return melhor_r2, melhor_ordem, melhor_eqm

if __name__ == '__main__':
    x = np.array([0, 3, -1, 4, 3, 5, 2, 10, 2, 4])
    y = np.array([0.2, 0.8, 2.4, 6.5, 7.1, 7.5, 7.7, 8.1, 8.9, 10.2])

    melhor_r2, melhor_ordem, melhor_eqm = ordenacao(x, y)

    print(f"Melhor ordem do polinômio: {melhor_ordem}")
    print(f"Melhor R²: {melhor_r2:.2f}")
    print(f"Melhor EQM: {melhor_eqm:.2f}")