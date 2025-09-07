import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs=10000):
        """
        Assign hyperparameters and initialize weights, bias
        and loss array

        Args:
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight, self.bias = None, None
        self.losses = []

    def _compute_loss(self, y, y_pred):
        """
        Computes the mean square error loss

        Args:
            y (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats
        """
        return np.mean((y - y_pred) ** 2)
    
    def compute_gradients(self, X, y, y_pred):
        """
        Computes the gradients of the loss with respect to the weights and bias

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats

        Returns:
            grad_w (array<n>): a vector of floats
            grad_b (float): a scalar
        """
        grad_w = -(2 / X.shape[0]) * np.dot(X, (y - y_pred))
        grad_b = -(2 / X.shape[0]) * np.sum(y - y_pred)
        return grad_w, grad_b 

    def update_parameters(self, grad_w, grad_b):
        """
        Updates the weights and bias using the gradients

        Args:
            grad_w (array<n>): a vector of floats
            grad_b (float): a scalar
        """
        self.weight -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.weight = 0
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = self.weight * X + self.bias

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        return self.weight * X + self.bias





