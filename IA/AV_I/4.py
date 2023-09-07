import math

x = [10, 20]
a = [12, 18]
b = [11, 20]
r = [0, 0]

for i in range(len(a)):
    r[0] += pow(x[i] - a[i], 2)
    r[1] += pow(x[i] - b[i], 2)

r[0] = math.sqrt(r[0])
r[1] = math.sqrt(r[1])

print('X e A: {:.2f}'.format(r[0]))
print('X e B: {:.2f}'.format(r[1]))

print('X pertence a classe B!')