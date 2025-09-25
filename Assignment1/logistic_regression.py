import numpy as np

class LogisticRegression():
    
    def __init__(self, learning_rate, epochs):
        """
        Assign hyperparameters and initialize weights, bias
        and loss array

        Args:
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []

    def sigmoid(self, z):
        """
        Computes the sigmoid activation function

        Args:
            z (array<m>): a vector of floats
        """
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        """
        Computes the cross-entropy loss

        Args:
            y (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats
        """
        return np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))

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
        m = X.shape[0]

        grad_w = -(1 / m) * np.dot(X.T, (y - y_pred))
        grad_b = -(1 / m) * np.sum(y - y_pred)

        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        """
        Updates the weights and bias using the gradients

        Args:
            grad_w (array<n>): a vector of floats
            grad_b (float): a scalar
        """
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    """
    
    """
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            lin_model = np.dot(X, self.weights) + self.bias
            
            y_pred = self.sigmoid(lin_model)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = (y_pred > 0.5).astype(int)
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
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
        lin_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(lin_model)
        return (y_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Generates probability predictions without classification
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
        
        Returns:
            A length m array of floats in the range [0, 1]
        """
        
        lin_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(lin_model)