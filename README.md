# Perceptron

This is implementation of perceptron function.

# Class: Perceptron()

Default initialises a perceptron object.

Paramteres:
None

Return:
Perceptron object
(It can be used to fit, evaluate, and predict the model)

# method: fit()

Fits the training data on data and labels provided.

Paramteres:
train_data :- D\*N numpy darray.
D - represents dimensions of array.
N - represents the number of training examples.

train_labels :- 1\*N numpy array, representing labels of (+1, -1)

Return:
th - 1\*N numpy darray representing hypothesis
th0 - (1, ) numpy array representing offset.

# method: eval()

Evaluates the trained model on testing data.

Parameters:-

testing_data: D\*n numpy array.
D - represents dimensions of array.
n - represents the number of testing examples.

testing_labels :- 1\*n numpy array, representing labels of (+1, -1)

Return:
Returns traning efficiency on test set.

# method: plot()

plots the model output on given data and the acutal labels side-by-side.

Parameters:-

data: D\*n numpy array.
D - represents dimensions of array.
n - represents the number of examples.

input labels :- 1\*n numpy array, representing labels of (+1, -1)

Return:
None.
