import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.special import softmax
import sys
import time

class NeuralNetwork:
    def __init__(self, layer, epochs, batch_size, learning_rate, validation_rate):
        self.weight = []
        self.bias = []
        for i in range(len(layer) - 1):
            self.weight.append(np.mat(np.random.randn(layer[i], layer[i + 1]) * np.sqrt(1.0 / layer[i])))
            self.bias.append(np.mat(np.random.randn(layer[i + 1]) * np.sqrt(1.0 / layer[i])))
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_rate = validation_rate

    def train(self, train_image, train_label):
        loss_log = []
        train_accuracy_log = []
        validation_accuracy_log = []
        
        train_start = time.time()
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print("Epoch " + str(epoch) + " started.")
            shuffled_index = np.random.permutation(len(train_image))
            shuffled_image = train_image.iloc[shuffled_index]
            shuffled_label = train_label.iloc[shuffled_index]
            validate_image = shuffled_image[int(len(shuffled_image) * (1 - self.validation_rate)) - 1:]
            validate_label = shuffled_label[int(len(shuffled_label) * (1 - self.validation_rate)) - 1:]
            shuffled_image = shuffled_image[: int(len(shuffled_image) * (1 - self.validation_rate)) - 1]
            shuffled_label = shuffled_label[: int(len(shuffled_label) * (1 - self.validation_rate)) - 1]
            batch_num = int(len(shuffled_image) / self.batch_size)
            if len(shuffled_image) % self.batch_size:
                batch_num += 1
            
            loss_log_batch = []
            accuracy_log_batch = []
            for batch in range(batch_num):
                # batch_start = time.time()
                # print("Epoch " + str(epoch) + " batch " + str(batch) + " started.")
                # Generate batch
                start = batch * self.batch_size
                end = (batch + 1) * self.batch_size
                if end > len(shuffled_image):
                    end = len(shuffled_image)
                # (batch_size, number of features)
                batch_image = shuffled_image[start: end]
                batch_image = np.mat(batch_image.to_numpy())
                # (batch_size, 1)
                batch_label = shuffled_label[start: end]
                batch_label = batch_label.to_numpy()
                
                loss, accuracy = self.train_batch(batch_image, batch_label)
                loss_log_batch.append(loss)
                accuracy_log_batch.append(accuracy)
                
                # batch_stop = time.time()
                # print("Epoch " + str(epoch) + " batch " + str(batch) + " finished. Loss: " + str(loss) + " Accuracy: " + str(accuracy) + ". Time cost: " + str(batch_stop - batch_start))
            
            loss = np.mean(loss_log_batch)
            train_accuracy = np.mean(accuracy_log_batch)
            loss_log.append(loss)
            train_accuracy_log.append(train_accuracy)
            
            validate_image = np.mat(validate_image.to_numpy())
            validate_label = validate_label.to_numpy()
            _, outputs = self.forward_pass(validate_image)
            validation_accuracy = self.compute_accuracy(outputs[-1], validate_label)
            validation_accuracy_log.append(validation_accuracy)
            
            epoch_stop = time.time()
            print("Epoch " + str(epoch) + " finished. Loss: " + str(loss) + ". Train accuracy: " + str(train_accuracy) + ". Validate accuracy: " + str(validation_accuracy) + ". Time cost: " + str(epoch_stop - epoch_start))
            # print()
        train_stop = time.time()
        print("Train finished after " + str(train_stop - train_start) + " seconds.")
        return loss_log, train_accuracy_log, validation_accuracy_log

    def train_batch(self, batch_image, batch_label):
        inputs, outputs = self.forward_pass(batch_image)
        loss = self.compute_loss(outputs[-1], batch_label)
        accuracy = self.compute_accuracy(outputs[-1], batch_label)
        self.backward_propagation(inputs, outputs, batch_label)
        return loss, accuracy

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def sigmoid_derivative(self, X):
        sig = self.sigmoid(X)
        return np.multiply(sig, (1 - sig))

    def forward_pass(self, batch_image):
        inputs = []
        inputs.append(None)
        outputs = []
        outputs.append(batch_image)
        for i in range(len(self.weight)):
            inputs.append(outputs[i] * self.weight[i] + self.bias[i])
            if i == len(self.weight) - 1:
                outputs.append(softmax(inputs[i + 1], axis=1))
            else:
                outputs.append(self.sigmoid(inputs[i + 1]))
        return inputs, outputs

    def compute_loss(self, output, batch_label):
        loss = []
        for i in range(len(batch_label)):
            loss.append(-np.log(output[i, batch_label[i]]))
        return np.mean(loss)

    def compute_accuracy(self, output, batch_label):
        correct = 0
        predict = np.argmax(output, axis=1)
        for i in range(len(predict)):
            if predict[i] == batch_label[i]:
                correct += 1
        return correct / len(predict)

    def backward_propagation(self, inputs, outputs, batch_label):
        batch_label_one_hot = np.zeros((len(batch_label), 10))
        for i in range(len(batch_label)):
            batch_label_one_hot[i, batch_label[i]] = 1
        d_loss_input = outputs[-1] - batch_label_one_hot
        for i in range(len(self.weight) - 1, -1, -1):
            d_loss_weight = outputs[i].T * d_loss_input / len(batch_label)
            d_loss_bias = np.sum(d_loss_input, axis=0) / len(batch_label)
            if i > 0:
                # d_loss_input for the next iteration
                d_loss_input = np.multiply(d_loss_input * self.weight[i].T, self.sigmoid_derivative(inputs[i]))
            
            self.weight[i] -= self.learning_rate * d_loss_weight
            self.bias[i] -= self.learning_rate * d_loss_bias

    # def plot(self, loss_log, train_accuracy_log, validation_accuracy_log):
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel('Epochs')
    #     ax.set_ylabel('Accuracy & Loss')
    #     ax.set_title('Learning curve')
    #     ax.plot(np.arange(self.epochs), loss_log, label='Loss')
    #     ax.plot(np.arange(self.epochs), train_accuracy_log, label='Train accuracy')
    #     ax.plot(np.arange(self.epochs), validation_accuracy_log, label='Validate accuracy')
    #     ax.legend()
    #     plt.savefig('Learning curve.png')

if __name__ == "__main__":
    layer = [784, 512, 256, 10]
    train_image_path = sys.argv[1]
    train_label_path = sys.argv[2]
    test_image_path = sys.argv[3]
    train_image = pd.read_csv(train_image_path, header=None)
    train_label = pd.read_csv(train_label_path, header=None)
    epochs = 80
    batch_size = 32
    learning_rate = 0.01
    validation_rate = 0.1
    # if len(train_image) < batch_size:
    #     batch_size = len(train_image)
    neural_network = NeuralNetwork(layer, epochs, batch_size, learning_rate, validation_rate)
    loss_log, train_accuracy_log, validation_accuracy_log = neural_network.train(train_image, train_label)
    # neural_network.plot(loss_log, train_accuracy_log, validation_accuracy_log)
    
    test_image = pd.read_csv(test_image_path, header=None)
    test_image = np.mat(test_image.to_numpy())
    _, outputs = neural_network.forward_pass(test_image)
    predict = np.argmax(outputs[-1], axis=1)
    np.savetxt('test_predictions.csv', predict, fmt="%i", delimiter=",")

# python3 NeuralNetwork3.py train_image.csv train_label.csv test_image.csv