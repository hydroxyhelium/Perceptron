from tkinter import N
import numpy as np 

# This is inspired by MIT 6.036 course. 

def load_data(generator, N):
    train_data, train_labels = generator(N)
    return train_data, train_labels 


class Perceptron():
    def __init__():
        self.th = np.zeros((1,d))
        self.th0 = np.zeros(1) 
        print("Class member initialized")
    
    def fit(self, train_data, train_labels, T=None):

        if(not T):
            T = 100

        d,n = train_data.shape # shape of the training data.

        th = np.zeros((1,d))
        th0 = np.zeros(1)  

        for t in range(T):
            for i in range(n):
                classifier_output = (np.dot(th, np.array([train_data[:, i]])) + th0[0])
                if (train_labels[0][i]*classifier_output) <=0:
                    th[0] += train_labels[0][i]*(train_data[:][i])
                    th0[0] += train_labels[0][i]
        
        self.th = th
        self.th0 = th0
    
    def eval(self, test_data, test_labels):
        th = self.th 
        th0 = self.th0

        d, n = test_data.shape 

        correct = 0 
        total = n 

        for i in range(n): 
            data = test_data[:, i]
            output = np.dot(th,np.array([data]))

            if(test_labels[i]*(output)>0):
                correct += 1

        return (correct/total)*100





def main():
    data_labels, data_features = None, None 




