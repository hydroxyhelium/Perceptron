from tkinter import N
import numpy as np 
import matplotlib.pyplot as plt


# This is inspired by MIT 6.036 course. 

def load_data(generator, N):
    train_data, train_labels = generator(N)
    return train_data, train_labels 


class Perceptron():
    def __init__(self):
        self.th0 = None
        self.th = None
        print("Class member initialized")
    
    def fit(self, train_data, train_labels, T=None):

        if(not T):
            T = 100
        d,n = train_data.shape # shape of the training data.

        self.th = np.zeros((1,d))
        self.th0 = np.zeros(1) 

        th = np.zeros((1,d))
        th0 = np.zeros(1)  

        for t in range(T):
            for i in range(n):
                classifier_output = (np.dot(th[0], train_data[:, i]) + th0[0])
                
                ##  print((np.dot(th[0], train_data[:, i]) + th0[0])) 
                if (train_labels[0][i]*classifier_output) <=0:
                    th[0] += train_labels[0][i]*(train_data[:, i])
                    th0[0] += train_labels[0][i]
        
        self.th = th
        self.th0 = th0

        return th, th0
    
    def eval(self, test_data, test_labels):
        th = self.th 
        th0 = self.th0

        d, n = test_data.shape 

        correct = 0 
        total = n 

        for i in range(n): 
            data = test_data[:, i]
            output = np.dot(th[0],data)

            if(test_labels[0][i]*(output)>0):
                correct += 1


        print("the accuracy for the model is "+str((correct/total)*100)+ " %. For the testing data.")
        return (correct/total)*100

    
    def plot(self, test_data, test_labels):
        output = []
        d, n = test_data.shape


        figure, axis = plt.subplots(1, 2)


        th, th0 = self.th, self.th0
        for i in range(n):
            input_data = test_data[:, i]
            model_output = np.dot(th[0], input_data)+th0[0]
            x = np.array(input_data[0])
            y = np.array(input_data[1]) 

            axis[0].set_title("Model Prediction")
            axis[1].set_title("Actual Labels")
            if(model_output>0):
                
                axis[0].plot(x, y, 'bo')
                axis[1].plot(x, y, 'bo')

            else: 
                axis[0].plot(x, y, 'ro')
                axis[1].plot(x, y, 'ro')
        
        plt.show()

        


        # this will only work if the data is 2-Dimensional 

    
def main():
    N = 8
    x1 = np.resize(np.linspace(0, 10, N, endpoint=True), (1, N))
    y1 = np.resize(np.array([4 for i in range(N)]), (1, N))
    
    x2 = np.resize(np.linspace(0, 10, N, endpoint=False), (1, N)) 
    y2 = np.resize(np.array([-1 for i in range(N)]), (1, N))

    top = np.concatenate((x1,y1), axis=0)
    #print(top)
    bottom = np.concatenate((x2, y2), axis = 0)
    data = np.concatenate((top, bottom), axis=1)
    #print(bottom)
    
    x2 = x2+ 5

    label1 = np.resize(np.array([1 for i in range(N)]), (1, N))
    label2 = np.resize(np.array([-1 for i in range(N)]), (1, N))

    input_labels = np.concatenate((label1, label2), axis=1)

    perceptron = Perceptron() 

    #print(input_labels)
    perceptron.fit(data, input_labels,1)
    perceptron.eval(data, input_labels)
    perceptron.plot(data, input_labels) 



    # print(label1)


if __name__ == '__main__':
    main() 



