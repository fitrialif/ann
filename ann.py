import numpy as np
import os
import matplotlib.pyplot as plt
from opt_utils import *
from data_utils import *
from utils import *

np.random.seed(1)


class Model:
    """
    Module is a super class. It could be a single layer, or a multilayer perceptron.
    """
    
    def __init__(self):
        self.train = True
        return
    
    def forward(self, _input):
        pass
    
    def backward(self, _input, _gradOutput):
        pass
        
    def parameters(self):
        pass
    
    def training(self):
        self.train = True
        
    def evaluate(self):
        self.train = False


class ANN(Model):
    """
    Artificial Neural Network base upon which we'll add layers
    """
    def __init__(self):
        Model.__init__(self)
        self.layers = []
        
    def get_size(self):
        self.size = len(self.layers)
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forprop(self, inputs):
        """
        Forward propagation: propagate through self.layers list, passing output from [i] to [i+1]
        """
        self.inputs = [inputs]
        self.get_size()
        n = self.size
        for i in range(n):
            prev_layer = self.layers[i]
            next_input = prev_layer.forprop(self.inputs[i])
            self.inputs.append(next_input)
        
        self.outputs = self.inputs[-1]
        return self.outputs
    
    def backprop(self, inputs, gradOut):
        """
        Backward propagation: propagate through self.gradIn list, passing output from [i] to [i-1]
        """
        self.get_size()
        n = self.size
        self.gradIn = [None] * (n + 1)
        self.gradIn[n] = gradOut

        for i in range(n, 0, -1):
            prev_layer = self.layers[i-1]
            prev_input = self.inputs[i-1]
            next_gradOut = prev_layer.backprop(prev_input, self.gradIn[i])
            self.gradIn[i-1] = next_gradOut
            
        gradIn = self.gradIn[0]
        return gradIn
    
    def parameters(self):
        params = []
        gradParams = []
        for layer in self.layers:
            ps, gPs = layer.parameters()
            if ps:
                params.append(ps)
                gradParams.append(gPs)
        return params, gradParams 


class Linear(Model):
    """
    Linear activation layer (fully connected)
    """
    def __init__(self, inSize, outSize, tanh=True):
        """
        inSize: number of units of previous layer (e.g., 784 for first hidden layer)
        outSize: number of (hidden) units in current layer (e.g., 300 for first hidden layer)
        """
        Model.__init__(self)
        if tanh:
            self.W = xavier_init((inSize, outSize))
            self.b = xavier_init((inSize, outSize), bias=True)
            self.dW = np.zeros((inSize, outSize))
            self.db = np.zeros(outSize)
        return
    
    def forprop(self, inputs):
        """
        Input:
        inputs: x    (n x inSize)
        
        Compute:
        z = xW + b   (n x inSize) * (inSize x outSize) + (1(n) x outSize)
        
        Output:
        outputs: z   (n x outSize)  
        """
        self.outputs = np.dot(inputs, self.W) + self.b
        return self.outputs
    
    def backprop(self, inputs, gradOut):
        """
        Input:
        inputs: x                     (n x inSize)
        gradOut: dL/dz                (n x outSize)
        
        Compute:
        dL/dW = (dz/dW).T * dL/dz     (inSize x n) * (n x outSize)
        dL/db = dL/dz * dz/db         (n x outSize) * (outSize x 1(n))
        dL/dx = dL/dz * dz/dx         (n x outSize) * (outSize x inSize)
        
        Output:
        self.dW: dL/dW                (inSize x outSize) - Not returned
        self.db: dL/db                (n x 1(n)) - Not returned
        gradIn: dL/dx                 (n x inSize)
        """
        n = inputs.shape[0]
        
        self.dW = np.dot(inputs.T, gradOut)
        self.db = np.dot(np.ones(n), gradOut)
        
        self.gradIn = np.dot(gradOut, self.W.T)
        return self.gradIn
        
    def parameters(self):
        return [self.W, self.b], [self.dW, self.db]


class Tanh(Model):
    """
    Tanh activation layer
    """
    def __init__(self):
        Model.__init__(self)
        return
    
    def forprop(self, inputs):
        """
        Input:
        inputs: z      (n x inSize)
        
        Compute:
        a = tanh(z)    (n x inSize)
        
        Output:
        output: a      (n x inSize)
        """
        self.output = np.tanh(inputs)
        return self.output
        
    def backprop(self, inputs, gradOut):
        """
        Input:
        inputs: z                (n x inSize)
        gradOut: dL/da           (inSize x outSize)
        
        Compute:
        dL/dz = da/dz * dL/da    (n x inSize) * (inSize x outSize) 
        
        Output:
        gradIn: dL/dz            (n x outSize)
        """
        self.gradIn =  gradOut * (1 - self.output * self.output)
        return self.gradIn
    
    def parameters(self):
        return None, None


class Softmax(Model):
    """
    Softmax output layer
    """
    def __init__(self):
        return
    
    def forprop(self, inputs, targets):
        """
        Input:
        inputs: z                                             (n x outSize)
        targets: y                                            (n x outSize)
        
        Compute:
        y_hat = exp(z_i - max(z)) / sum(exp(z_i - max(z)))    (n x outSize)
        Loss = - sum(y_i * log(y_hat))                        (scalar)
        
        Output:
        probs: y_hat                                          (n x outSize) - Not returned
        output: Loss                                          (scalar)
        """
        inputs_stable = inputs - inputs.max(axis=1, keepdims=True)
        e = np.exp(inputs_stable)
        self.probs = e / e.sum(axis=1, keepdims=True)
        
        logs = np.multiply(targets, np.log(self.probs))
        L_i = np.max(-logs, axis=1)
        Loss = np.mean(L_i)

        self.output = Loss
        return self.output
        
    def backprop(self, inputs, targets):
        self.gradIn = (self.probs - targets) / inputs.shape[0]
        return self.gradIn
    
    def parameters(self):
        return None, None


class Dropout(Model):
    """
    Dropout layer to be placed upon another activation (could be linear) or input
    """
    def __init__(self, prob = 0.5):
        Model.__init__(self)
        self.prob = prob #self.p is the drop rate, if self.p is 0, then it's a identity layer
        
    def forprop(self, inputs):
        self.output = inputs
        self.dropout = np.random.binomial(1, self.prob, size=inputs.shape)
        if self.train:
            self.output *= self.dropout
        else:
            self.output *= self.prob
        return self.output
    
    def backprop(self, inputs, gradOut):
        self.gradIn = gradOut
        if self.train:
            self.gradIn *= self.dropout
        else:
            self.gradIn *= self.prob
        return self.gradIn
    
    def parameters(self):
        """
        No trainable parameters.
        """
        return None, None


#class BatchNorm(Model):
#    def __init__(self):
#        Model.__init__(self)
#        self.mu = []
#        self.sig = []
#        return
#    
#    def normalize(self, inputs, mu, sig):
#        return (inputs - mu) / sig
#    
#    def forprop(self, inputs):
#        if self.train:
#            self.mu = np.mean(inputs)
#            self.sig = np.std(inputs) + 1e-8
#            #self.mu.append(np.mean(inputs, axis=1, keepdims=True))
#            #self.sig.append(np.std(inputs, axis=1, keepdims=True) + 1e-8)
#
#            self.output = self.normalize(inputs, self.mu, self.sig)
#        else:
#            mu_overall = np.mean(self.mu, axis=0, keepdims=True)
#            sig_overall = np.mean(self.sig, axis=0, keepdims=True)
#            
#            self.output = self.normalize(inputs, mu_overall, sig_overall)
#        
#    def backprop(self, inputs, gradOutput):
#        self.gradIn = gradOutput * self.sig
#        
#    def parameters(self):
#        return None, None


def error(X, y, model):
    model.evaluate()
    pred = model.forprop(X).argmax(-1) == y.argmax(-1)
    err = 100 * (1 - pred.mean())
    model.training()
    return err

def train(X_train, y_train, model, smloss, X_val, y_val, X_test, y_test, LR, batch_size=50, RMS_rho=None, verbose=False):
    """
    Run the train + evaluation on a given train/val partition
    trainopt: various (hyper)parameters of the training procedure
    During training, choose the model with the lowest validation error. (early stopping)
    """
    n = X_train.shape[0]
    epoch = 1
    val_err_per_epoch = []
    val_err = []
    test_err = []
    r = 0
    
    while True:
        for i in range(0, n, batch_size):
            X_mini = X_train[i:i+batch_size]
            y_mini = y_train[i:i+batch_size]

            preds = model.forprop(X_mini)
            loss = smloss.forprop(preds, y_mini)
            #print(loss)
            dloss = smloss.backprop(preds, y_mini)
            model.backprop(X_mini, dloss)

            params, gradParams = model.parameters()
            if RMS_rho is not None:
                RMSProp(params, gradParams, RMS_rho, 0.01, r)
            else:
                SGD(params, gradParams, 0.1, 0.0005)
            
            if verbose:
                if i % 4000 == 0:            
                    tr_err = error(X_train, y_train, model)
                    v_err = error(X_val, y_val, model)
                    te_err = error(X_test, y_test, model)
                    print('{:8} batch loss: {:.3f} train error: {:.3f} val error: {:.3f} test error: {:.3f}'.format(i, loss, tr_err, v_err, te_err))    

        epoch += 1
        val_err.append(error(X_val, y_val, model))
        test_err.append(error(X_test, y_test, model))
        if len(val_err) > 1:
            if val_err[-1] > val_err[-2]:
                print('Done! \t Final validation error: {:.3f} \t Final test error: {:.3f}'.format(val_err[-1], test_err[-1]))
                return (model, val_err, test_err)

