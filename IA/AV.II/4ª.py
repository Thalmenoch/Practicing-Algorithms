import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def ordenacao(x, y):
    melhor_ordem
    melhor_r2
    melhor_eqm
    for ordem in range(1, 10):  # Tente polinômios de ordem 1 a 10
        coeficientes = np.polyfit(x, y, ordem)
        y_previsto = np.polyval(coeficientes, x)
    
        r2 = r2_score(y, y_previsto)
        eqm = mean_squared_error(y, y_previsto)
        
        if r2 > melhor_r2:
            melhor_r2 = r2
            melhor_ordem = ordem
            melhor_eqm = eqm
        
        return melhor_r2, melhor_ordem, melhor_eqm

if __name__ == '__main__':
    x = np.array([0, 3, -1, 4, 3, 5, 2, 10, 2, 4])
    y = np.array([0.2, 0.8, 2.4, 6.5, 7.1, 7.5, 7.7, 8.1, 8.9, 10.2])