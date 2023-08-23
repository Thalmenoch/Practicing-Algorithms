a = [10, 1, 3] # ponto A
b = [4, 5, 4] # ponto B
r = 0

for i in range(len(a)):
   r += abs(a[i] - b[i]) # aqui é feito a soma das diferenças absolutas, utilizando da função abs para calcular valor absoluto.  

print('distancia de manhattan entre estes dois pontos: ',r)