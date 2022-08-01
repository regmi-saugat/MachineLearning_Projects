import numpy as np
class SVM_Classifier():
    #@ Initiating the hyperparameter
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
    #@ Fitting the dataset to SVM Classifier
    def fit(self, X, Y):
        self.m, self.n = X.shape                                                                    # m --> data points & n --> input features
        #@ Initiating weights and bias value
        self.w = np.zeros(self.n)                                                                   # Initializing weights
        self.b = 0                                                                                  # Initializing bias
        self.X = X                                                                                  # X --> Input feature
        self.Y = Y                                                                                  # Y --> Output feature that we are going to predict
        #@ Implementing Gradient Descent algorithm for Optimization
        for i in range(self.no_of_iterations):
            self.update_weights()
    #@ Function for updating the weight and bias value
    def update_weights(self):
        y_label = np.where(self.Y <= 0, -1, 1)                                                      # Label Encoding
        # Building Gradient (dw & db)
        for index, x_i in enumerate(self.X):
            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
        if (condition == True):
            dw = 2 * self.lambda_parameter * self.w
            db = 0
        else:
            dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
            db = y_label[index]
        #@ Updating Gradient Descent
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
    #@ Predict the label for given input value
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)                                             # Final predicted output
        return y_hat








        







