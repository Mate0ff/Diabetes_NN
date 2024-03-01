from turtle import mode
import pandas as pd
import numpy as np


"""
This task involves creating a neural network model that, based on various predictors, will determine whether a given patient has diabetes or not.
In summary, we will develop a neural network that classifies patients into binary categories based on the presence or absence of diabetes.

"""


class NeuralNetwork:

    np.random.seed(42)  # Seed for reproducibility

    def __init__(self,
                 data: pd.DataFrame,
                 epochs: int = 10_000,
                 learning_rate: float = 0.01,
                 number_ofNeurons: list[int] = [5, 5, 5],
                 number_ofLayers: int = 3,
                 size_ofTrainSet: float = 0.7,
                 activation_function: str = 'sigmoid'
                 ) -> None:
        # Here are all arguments that the neural network will use across algorithm
        self.size_ofTrainSet = self._validate_size_ofTrainSet(size_ofTrainSet)
        self.X_train, self.X_test, self.Y_train, self.Y_test = self._validate_data(data)

        self.input_neurons: int = len(data.columns) - 1  # Number of variables - 1
        self.output_nuerons = 1
        if len(number_ofNeurons) != number_ofLayers:
            raise Exception("Incorrect parameters")
        self.layer_neurons = number_ofNeurons
        self.epochs = epochs
        self.learnig_rate = learning_rate
        self.activation_function = activation_function
        
        self.weights = None
        self.biases = None
        self.y_pred = None


    def _validate_data(self, data: pd.DataFrame):
        try:
            X = data.drop('Outcome', axis=1).values
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardize the features
            Y = data['Outcome'].values.reshape(-1, 1)  

            # Split Data into Training and Test Sets
            split_ratio = self.size_ofTrainSet
            split_index = int(split_ratio * len(data))
            X_train, X_test = X[:split_index], X[split_index:]
            Y_train, Y_test = Y[:split_index], Y[split_index:]

            return X_train, X_test, Y_train, Y_test
        except:
            raise Exception("Something happened when validating data")

    def _validate_size_ofTrainSet(self, value: float) -> float:
        try:
            if not 0 <= value <= 1:
                raise ValueError("Size of training set must be a float between 0 and 1.")
            return value
        except:
            raise Exception("Something happened when validating data")


    def train_network_sigmoidActivation(self) -> None:
        np.random.seed(42)  # Seed for reproducibility

        # Get the class variables here
        X_train, y_train, input_neurons, layer_neurons, output_neurons, epochs, learning_rate, activation_function = self.X_train, self.Y_train, self.input_neurons, self.layer_neurons, self.output_nuerons, \
                                                                                                self.epochs, self.learnig_rate, self.activation_function
        # Initialize weights and biases for each layer
        weights = [np.random.rand(input_neurons, layer_neurons[0])]
        biases = [np.zeros((1, layer_neurons[0]))]

        for i in range(len(layer_neurons) - 1):
            weights.append(np.random.rand(layer_neurons[i], layer_neurons[i + 1]))
            biases.append(np.zeros((1, layer_neurons[i + 1])))

        weights.append(np.random.rand(layer_neurons[-1], output_neurons))
        biases.append(np.zeros((1, output_neurons)))

        # Training loop
        for epoch in range(epochs):
            # Forward propagation through the network
            activations = [X_train]

            for w, b in zip(weights, biases):
                activations.append(activate_function(np.dot(activations[-1], w) + b,activation_function))

            # Compute error and cost (Mean Squared Error)
            error = activations[-1] - y_train
            cost = np.mean(np.square(error))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")

            # Backward propagation to adjust weights and biases
            for i in reversed(range(len(weights))):
                d_activation = activate_function_derative(activations[i + 1],activation_function)
                error = error * d_activation
                delta = np.dot(activations[i].T, error)
                weights[i] -= learning_rate * delta
                biases[i] -= learning_rate * np.mean(error, axis=0, keepdims=True)
                error = np.dot(error, weights[i].T)

        self.weights, self.biases = weights, biases

    def predict(self) -> None:
        X_test = self.X_test
        activations = X_test
        for w, b in zip(self.weights, self.biases):
            activations = activate_function(np.dot(activations, w) + b,self.activation_function)
        
        if self.activation_function == 'sigmoid' or self.activation_function == 'relu' or self.activation_function == 'softmax': 
            self.y_pred = activations >= 0.5  # Classify as 1 if activation >= 0.5 else 0
        elif self.activation_function == 'tanh':
            self.y_pred = activations >= 0  # Classify as 1 if activation >= 0 else 0

    def show_accuracy(self) -> None:
        y_true, y_pred = self.Y_test, self.y_pred

        # Calculate True Positive(1,1), False Positivie(0,1), False Negative(0,0), True Negative(1,0)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        # Calculate Accuracy, Recall, Precision, and F1 Score
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1_score:.2f}")


"""
Activation functions
"""

def activate_function(x, activation_function):
    if activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation_function == 'tanh':
        return np.tanh(x)
    elif activation_function == 'relu':
        return np.where(x > 1, 1, np.maximum(0,x))
    elif activation_function =='softmax':
        return np.exp(x) / np.exp(x).sum()

    return 0    


def activate_function_derative(x, activation_function):
    if activation_function == 'sigmoid':
        return x * (1 - x)
    elif activation_function == 'tanh':
        return 1.0 - np.tanh(x)**2
    elif activation_function == 'relu':
        return np.where(x <= 0, 0, 1)
    elif activation_function =='softmax':
        s = activate_function(x,activation_function)
        return s * (1 - s)
       
    return 0 




def main() -> None:
    """
        Here is the place to ONLY start the model with different parameters
    """
    # Load and Prepare Data
    data = pd.read_csv("...\\diabetes.csv")

    # Options for actiavtion functions: Sigmoid, tanh, relu, softmax
    model = NeuralNetwork(data=data,
                          epochs=20_000,
                          learning_rate=0.001,
                          number_ofNeurons=[2,2],
                          number_ofLayers=2,
                          size_ofTrainSet=0.5,
                          activation_function='sigmoid')

    model.train_network_sigmoidActivation()
    model.predict()
    print(model.y_pred)
    model.show_accuracy()


if __name__ == '__main__':
    main()
