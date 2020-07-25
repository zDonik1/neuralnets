import mnist_loader as ml
import neuralnetwork as nn

training_data, validation_data, test_data = ml.load_data_wrapper()
net = nn.NeuralNetwork([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
