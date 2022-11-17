import numpy as np
import pandas as pd
from numpy import random
import scipy.optimize as sopt



class nnetwork:
    def __init__(self,x_data, y_data, ldims, W):
        self.ldims = np.array(ldims,dtype=int)
        self.nlayers = self.ldims.size-1
        self.wsize = np.dot(self.ldims[:-1],self.ldims[1:])
        assert W.size == self.wsize
        assert W.dtype == np.float64
        assert x_data.ndim == 2 and y_data.ndim == 2
        assert x_data.shape[0] == ldims[0]
        assert y_data.shape[0] == ldims[-1]
        self.ndata = x_data.shape[1]
        self.x_data = x_data
        self.y_data = y_data
        self.W = W.reshape([self.wsize,])
        self.gradW = np.zeros([self.wsize,],dtype=np.float64)
        self.gradWshape = np.ndarray(shape=[self.nlayers,],dtype=object)
        self.Wp = np.ndarray(shape=[self.nlayers,],dtype=object)
        self.layer_result = np.ndarray(shape=[self.nlayers,],dtype=object)
        self.layer_result_aux = np.ndarray(shape=[self.nlayers,],dtype=object)
        begin = 0
        for i in range(self.nlayers):
            self.layer_result[i] = np.ndarray(shape=[self.ldims[i+1],],dtype=np.float64)
            self.layer_result_aux[i] = np.ndarray(shape=[self.ldims[i+1],],dtype=np.float64)
            end = begin + self.ldims[i]*self.ldims[i+1]
            self.gradWshape[i] = np.array([begin,end],dtype=int)
            self.Wp[i] = self.W[begin:end].reshape(self.ldims[i+1],self.ldims[i])
            begin = end

    def get_layer_weights(self, W, k):
        assert k >= 0 and k < self.nlayers
        self.W = W
        return self.Wp[k].reshape(-1)

    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))

    def dsigmoid(self,x):
        z = np.exp(-x)
        return z / ((1.0 + z)**2)

    def make_layer_result(self, x):
        self.layer_result[0] = self.Wp[0] @ x
        for i in range(1,self.nlayers):
            self.layer_result[i] = self.Wp[i] @ self.sigmoid(self.layer_result[i-1])

    def make_layer_result_from(self, x, k):
        assert k >= 0 and k < self.nlayers
        if k==0:
            self.make_layer_result(x)
        else:
            for i in range(k,self.nlayers):
                self.layer_result[i] = self.Wp[i] @ self.sigmoid(self.layer_result[i-1])

    def make_layer_result_aux(self, x, k = 0):
        assert k >= 0 and k < self.nlayers
        self.layer_result_aux[-1] = self.dsigmoid(self.layer_result[-1])

        for i in reversed(range(k,self.nlayers-1)):
            self.layer_result_aux[i] = (self.layer_result_aux[i+1] @ (self.Wp[i+1] * self.dsigmoid(self.layer_result[i]))).reshape(-1)

    def forward(self, x):
        self.make_layer_result(x)
        return self.sigmoid(self.layer_result[-1])

    def forward_from(self, x, k):
        assert k >= 0 and k < self.nlayers
        self.make_layer_result_from(x,k)
        return self.sigmoid(self.layer_result[-1])


    def residual(self, W, n):
        assert n >= 0 and n < self.ndata
        self.W = W
        return (self.forward(self.x_data[:,n]) - self.y_data[:,n]).item()

    def grad_residual(self, W, n):
        assert n >= 0 and n < self.ndata
        self.W = W.reshape(np.shape(self.W))

        self.make_layer_result(self.x_data[:,n])
        self.make_layer_result_aux(self.x_data[:,n],0)

        begin,end = self.gradWshape[0]
        self.gradW[begin:end] = (self.layer_result_aux[0].reshape(-1,1) * self.x_data[:,n]).reshape(-1)
        for i in range(1,self.nlayers):
            begin,end = self.gradWshape[i]
            self.gradW[begin:end] = (self.layer_result_aux[i].reshape(-1,1) * self.sigmoid(self.layer_result[i-1])).reshape(-1)
        return self.gradW

    def residual_and_grad_residual(self, W, n):
        assert n >= 0 and n < self.ndata
        self.W = W

        res = (self.forward(self.x_data[:,n]) - self.y_data[:,n]).item()
        self.make_layer_result_aux(self.x_data[:,n])

        begin,end = self.gradWshape[0]
        self.gradW[begin:end] = (self.layer_result_aux[0].reshape(-1,1) * self.x_data[:,n]).reshape(-1)
        for i in range(1,self.nlayers):
            begin,end = self.gradWshape[i]
            self.gradW[begin:end] = (self.layer_result_aux[i].reshape(-1,1) * self.sigmoid(self.layer_result[i-1])).reshape(-1)
        return res, self.gradW

    def residual_Wp(self, Wp, k, n):
        assert k >= 0 and k < self.nlayers
        assert n >= 0 and n < self.ndata

        self.Wp[k] = Wp.reshape(np.shape(self.Wp[k]))

        return (self.forward(self.x_data[:,n]) - self.y_data[:,n]).item()

    def grad_residual_Wp(self, Wp, k, n): # = d_r/d_W1
        assert n >= 0 and n < self.ndata
        assert k >= 0 and k < self.nlayers
        self.Wp[k] = Wp.reshape(np.shape(self.Wp[k]))

        self.make_layer_result(self.x_data[:,n])
        self.make_layer_result_aux(self.x_data[:,n],k)

        begin,end = self.gradWshape[k]
        if k == 0:
            self.gradW[begin:end] = (self.layer_result_aux[k].reshape(-1,1) * self.x_data[:,n]).reshape(-1)
        else:
            self.gradW[begin:end] = (self.layer_result_aux[k].reshape(-1,1) * self.sigmoid(self.layer_result[k-1])).reshape(-1)
        return self.gradW[begin:end]

    def residual_and_grad_residual_Wp(self, Wp, k, n): # = d_r/d_W1
        assert n >= 0 and n < self.ndata
        assert k >= 0 and k < self.nlayers
        self.Wp[k] = Wp.reshape(np.shape(self.Wp[k]))

        res = (self.forward(self.x_data[:,n]) - self.y_data[:,n]).item()
        self.make_layer_result_aux(self.x_data[:,n],k)

        begin,end = self.gradWshape[k]
        if k == 0:
            self.gradW[begin:end] = (self.layer_result_aux[k].reshape(-1,1) * self.x_data[:,n]).reshape(-1)
        else:
            self.gradW[begin:end] = (self.layer_result_aux[k].reshape(-1,1) * self.sigmoid(self.layer_result[k-1])).reshape(-1)
        return res, self.gradW[begin:end]

    def fun(self,W):
        val = np.float64(0.0)
        for i in range(self.ndata):
            val += self.residual(W,i)**2
        return 0.5*val

    def grad_fun(self,W):
        grad = np.zeros(W.shape)
        for i in range(self.ndata):
            res, grad_res = self.residual_and_grad_residual(W,i)
            grad += res*grad_res
        return grad

    def fun_and_grad_fun(self,W):
        val = np.float64(0.0)
        grad = np.zeros(W.shape)
        for i in range(self.ndata):
            res, grad_res = self.residual_and_grad_residual(W,i)
            grad += res*grad_res
            val += res**2
        return 0.5*val, grad

    def fun_Wp(self,Wp,k):
        val = np.float64(0.0)
        for i in range(self.ndata):
            val += self.residual_Wp(Wp,k,i)**2
        return 0.5*val

    def grad_fun_Wp(self,Wp,k):
        grad = np.zeros(Wp.shape)
        for i in range(self.ndata):
            res, grad_res = self.residual_and_grad_residual_Wp(Wp,k,i)
            grad += res*grad_res
        return grad

    def fun_and_grad_fun_Wp(self,Wp,k):
        val = np.float64(0.0)
        grad = np.zeros(Wp.shape)
        for i in range(self.ndata):
            res, grad_res = self.residual_and_grad_residual_Wp(Wp,k,i)
            grad += res*grad_res
            val += res**2
        return 0.5*val, grad

    def residuals_array(self,W):
        res = np.ndarray(shape=[self.ndata,],dtype=np.float64)
        for i in range(self.ndata):
            res[i] = self.residual(W,i)
        return res

    def jacobian(self,W):
        jac = np.ndarray(shape=[self.ndata,self.wsize],dtype=np.float64)
        for i in range(self.ndata):
            jac[i,:] = self.grad_residual(W,i).reshape(1,-1)
        return jac

    def residuals_array_Wp(self,Wp,k):
        assert k >= 0 and k < self.nlayers
        res = np.ndarray(shape=[self.ndata,],dtype=np.float64)
        for i in range(self.ndata):
            res[i] = self.residual_Wp(Wp,k,i)
        return res

    def jacobian_Wp(self,Wp,k):
        assert k >= 0 and k < self.nlayers
        begin,end = self.gradWshape[k]
        jac = np.ndarray(shape=[self.ndata,end-begin],dtype=np.float64)
        for i in range(self.ndata):
            jac[i,:] = self.grad_residual_Wp(Wp,k,i).reshape(1,-1)
        return jac

def minimize(nn, layer='all', method='BFGS', hess_type='BFGS', tol=None, callback=None, options=None):
    """
    :param method: method used by the optimizer, it should be one of:
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'trust-constr',
        'dogleg',  # requires positive semi definite hessian
        'trust-ncg',
        'trust-exact', # requires hessian
        'trust-krylov'
    """

    if 'least_square' in method:
        if layer == 'all':
            fun = nn.residuals_array
            x_init = nn.W
            grad_fun = nn.jacobian
            args = ()
        elif isinstance(layer,int):
            assert layer >= 0 and layer < nn.nlayers
            x_init = nn.get_layer_weights(nn.W,layer)
            fun = nn.residuals_array_Wp
            grad_fun = nn.jacobian_Wp
            args = (layer,)
        else:
            raise ValueError

        if 'lm' in method:
            ls_method = 'lm'
            ls_loss = 'linear'
        elif 'trf' in method:
            ls_method = 'trf'
            ls_loss = 'soft_l1'
            # ls_loss = 'cauchy'
            # ls_loss = 'huber'
        elif 'dogbox' in method:
            ls_method = 'dogbox'
            ls_loss = 'soft_l1'
        else:
            raise ValueError

        optim_res = sopt.least_squares(fun, x_init, jac=grad_fun, method=ls_method, loss=ls_loss, xtol=tol, args=args)
    else:
        if layer == 'all':
            x_init = nn.W
            fun = nn.fun
            grad_fun = nn.grad_fun
            args = ()
        elif isinstance(layer,int):
            assert layer >= 0 and layer < nn.nlayers
            x_init = nn.get_layer_weights(nn.W,layer)
            fun = nn.fun_Wp
            grad_fun = nn.grad_fun_Wp
            args = (layer)
        else:
            raise ValueError
        hess = sopt.BFGS(exception_strategy='damp_update') if hess_type=='BFGS' else sopt.SR1 if hess_type=="SR1" else None

        optim_res = sopt.minimize(fun, x_init, args, method=method, jac=grad_fun,
                                  hess=hess if method in ['Newton-CG', 'trust-ncg','trust-krylov','trust-constr','dogleg','trust-exact'] else None,
                                  tol=tol, callback=callback, options=options)
    return optim_res


# lendo dados
dados = pd.read_excel('dados.xls')

#definindo entrada e saida
x = np.array(dados.iloc[:,0:-1].T,dtype=np.float64)
y = np.array([dados.iloc[:,-1].T],dtype=np.float64)

# layers
layers = [3,10,1]

def size_from_layers(ld):
    return sum([ld[i]*ld[i+1] for i in range(len(ld)-1)])

#inicializando pesos
np.random.seed(1)
W = np.array(np.random.rand(size_from_layers(layers),),dtype=np.float64)

nn=nnetwork(x,y,layers,W)
cost = nn.fun(nn.W)

print('Network size: ', size_from_layers(layers))
print('Initital cost: ', cost)

epsilon = 1.0e-3
previous_cost = 0
N=0
nfeval = 0
njaceval = 0

# Métodos gerais de minimização:
# method = 'BFGS'
# method = 'SLSQP'
# method = 'L-BFGS-B'
# method = 'trust-ncg'
# method = 'trust-exact'  - precisa de hessiana (não funciona com atualização BFGS)
# method = 'dogleg'  - precisa de hessiana (não funciona com atualização BFGS)
# method = 'Newton-CG'
hess='BFGS'

# Mínimos quadrados:
#method = 'least_square_lm'
method = 'least_square_trf'
# method = 'least_square_dogbox'
while abs(cost-previous_cost) > epsilon*cost:
    previous_cost = cost
    N+=1
    print( 'Iteração: ', N)
    for i in reversed(range(nn.nlayers)):
        res = minimize(nn, layer=i, method=method, hess_type=hess, tol=epsilon)

        cost = res.cost if 'least_square' in method else res.fun

        nfeval += res.nfev
        njaceval += res.njev
        print('             Erro total: ', cost, ', nfeval: ', nfeval, ', njacval: ', njaceval)

print( 'Resultado:','\n   Camadas: ', layers, ', Número de variáveis: ', size_from_layers(layers),'\n   Método: ', method, '\n   Total iterações: ', N, '\n   Erro final: ', cost, '\n   nfeval: ', nfeval, '\n   njacval: ', njaceval)
