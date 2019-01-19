
import numpy as np
import time
import random
def sigmoid(x):
    return 1/(1+np.exp(-x))


class net:
    def __init__(self):
        pass
    def __init__(self,input_n,hidden_n,output_n):
        self.N_input = input_n
        self.N_hidden = hidden_n
        self.N_output = output_n
        self.weights_in_hidden = np.random.uniform(low=-1, high=1, size=(self.N_input,self.N_hidden))
        self.weights_hidden_out = np.random.uniform(low=-1, high=1, size=(self.N_hidden,self.N_output))
    def feed(self,X):
        hidden_layer_in = np.dot(X, self.weights_in_hidden)
        hidden_layer_out = sigmoid(hidden_layer_in)
        output_layer_in = np.dot(hidden_layer_out, self.weights_hidden_out)
        o = sigmoid(output_layer_in)
        if o[0] > o[1] :
            return 0
        return 1

    def copy(self):
        n=copy.deepcopy(self)
        return n

    def mutate(self,value):
        hap=0;
        for i in range(len(self.weights_in_hidden)):
            for j in range(len(self.weights_in_hidden[i])):
                if np.random.normal()<value:
                    self.weights_in_hidden[i][j]+=random.gauss(0,0.1)
                    hap+=1
        for i in range(len(self.weights_hidden_out)):
            for j in range(len(self.weights_hidden_out[i])):
                if np.random.normal()<value:
                    self.weights_hidden_out[i][j]+=random.gauss(0,0.1)
                    hap+=1
