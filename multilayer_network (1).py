import numpy as np
import pandas as pd
from numpy import random

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def dsigmoid(x):
    z = np.exp(-x)
    return z / ((1.0 + z)**2)

# two layer network
def residual(W1, W2, x, d):
    return (sigmoid(W2 @ sigmoid(W1 @ x)) - d).reshape(-1)

def grad_residual(W1, W2, x): # = [d_r/d_W1; d_r/d_W2]
    l1 = W1 @ x.reshape(-1,1)
    sl1 = sigmoid(l1)
    ds = dsigmoid(W2 @ sl1)
    return np.concatenate([np.reshape(ds * (W2.reshape(-1,1) * dsigmoid(l1)) @ x.reshape(1,-1),(-1,1)),\
                           ds * sl1])

def grad_residual_W1(W1, W2, x): # = d_r/d_W1
    l1 = W1 @ x.reshape(-1,1)
    ds = dsigmoid(W2 @ sigmoid(l1))
    return ds * (W2.reshape(-1,1) * dsigmoid(l1)) @ x.reshape(1,-1)

def grad_residual_W2(W1, W2, x): # = d_r/d_W2
    sl1 = sigmoid(W1 @ x.reshape(-1,1))
    ds = dsigmoid(W2 @ sl1)
    return ds * sl1

def obj(r):
    return (0.5 * np.sum(r**2))


# lendo dados
dados = pd.read_excel('dados.xls')

#definindo entrada e saida
x = np.array(dados.iloc[:,0:-1].T,dtype=np.float64)
y = np.array(dados.iloc[:,-1],dtype=np.float64)

#inicializando pesos
np.random.seed(1)

W = np.array(np.random.rand(40,1),dtype=np.float64)
W1 = W[0:30].reshape(10,3)
W2 = W[30:40].reshape(1,10)

# treinando a rede

epslon = 1.0e-03 # relative error
l_rate = 0.3     # learning rate

previous_cost = np.float64(0.0)
r = residual(W1,W2,x,y)
cost = obj(r)
print(cost)

N=0
while abs(cost-previous_cost) > epslon*previous_cost:
    for i in range(200):
        # grad_res += r[i]*grad_residual(W1,W2,x[:,i])
        r[i] = residual(W1,W2,x[:,i],y[i])
        # W1 -= l_rate * r[i]*grad_residual_W1(W1,W2,x[:,i])
        W2 -=  l_rate * r[i]*grad_residual_W2(W1,W2,x[:,i]).reshape(1,-1)

    for i in range(200):
        # grad_res += r[i]*grad_residual(W1,W2,x[:,i])
        r[i] = residual(W1,W2,x[:,i],y[i])
        W1 -= l_rate * r[i]*grad_residual_W1(W1,W2,x[:,i])
        # W2 -= l_rate * r[i]*grad_residual_W2(W1,W2,x[:,i]).reshape(1,-1)

    # W -= l_rate * grad_res
    r = residual(W1,W2,x,y)
    previous_cost = cost
    cost = obj(r)
    if N % 10 == 0:
        print(cost,'Erro total, iteracao:',N)
    N += 1

print(cost,'Erro total, iteracao:',N)
print(N,'epocas')
