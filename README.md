# Multilayer-Perceptron-Classification

functions implemented in utils.py:

Activation Functions:

Identity Function : this function returns the same input array anf if the derivative is applied it returns the array of ones of same size of input array

sigmoid Function : this function returns the 1/(1 + np.exp(-x)) (this put the values with in range[0,1]), and the derivative returns (1/(1 + np.exp(-x)))*(1-(1/(1 + np.exp(-x)))) 

tanh Function: this function returns the np.tanh(x) (this put the values with in range[-1,1]), and the derivative returns 1.0 - np.tanh(x)**2

relu Function: this function returns 0 if x<0 else returns x , and if derivative is applied it returns 0 if x<0 else returns 1

cross_entropy: This function is used to calculate the error/ loss , it returns y-p 
where  y: A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input data used when fitting the model.
p: A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax output activation function.

one_hot_encoding: Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the original data.

Fit() function:

Initialize the h_weights randomly of size (no.of input layers, n_hidden)  and also initialize the the the h_bias to 1 initially.

Initialize the o_weights randomly of size (n_hidden,no. of output layers)  and also initialize the the the h_bias to 1 initially.

Forward Propogation:

 For each hidden layer inputs are combined with the initial weights in a weighted sum and bias is added then subjected to the activation function. 

 Then the outputs of each hidden layer is passed to the output layer and then for each output layer inputs are combined with the initial weights in a weighted sum and bias is added then subjected to the softmax function and returns the predicted probabilities of each output. 

Backward Propogation:

The gradient of the Mean Squared Error is computed over all input and output pairs in each iteration. The weights of the hidden layer are then modified with the gradient value to propagate it back. That is how the weights are transmitted back to the neural network's starting point.

all weight between hidden and output layer are updated as follows:
W−=learning_rate⋅(δ − Output_hiddenlayer)
δ: error in the output layer

similarly the weights between the input and the output layer are updated.

for each train data point the above steps are performed for the given n_iterations, and breaks the iterations where there is no difference in mean_squared_error of current and previous iterations.

Predict() function:

 for each data point in test data only forward propogation is performed with the updated weights after training the network.

 and the maximum probability in the output layer is predicted as the class lable of each test data point. 

References:
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
https://cs230.stanford.edu/files/C1M3.pdf
