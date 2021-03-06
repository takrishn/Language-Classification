from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import time
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import unicodedata
import string
import pandas as pd
from collections import defaultdict

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The MNIST dataset is a built-in dataset in torchvision
batch_size = 100
train_dataset = torchvision.datasets.MNIST(root='../../data',
										   train=True,
										   transform=transforms.ToTensor(),
										   download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
										  train=False,
										  transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False)


# Create a simple neural network with just one hidden layer 
# and one Softmax layer between hidden layer and output.
class SimpleNeuralNet(nn.Module):

	def __init__(self, input_size, hidden_size, num_classes):
		super(SimpleNeuralNet, self).__init__()
		# fill in the declarations of the layers here
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)
		self.softmax = nn.Softmax()

	def forward(self, x):
		# fill the forward logic here
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.softmax(x)
		return x


def train_and_test_simple_net(input_size, hidden_size, num_classes):
	num_epochs = 5
	learning_rate = 0.001
	model = SimpleNeuralNet(input_size, hidden_size, num_classes).to(device)

	# PROBLEM 2.2 (10 points) Complete the Loss functions and optimizer
	# The loss function should be cross-entropy loss because it's a multi-class 
	# problem. The optimizer used should be Adam and should be bound with the neural net 
	# "model". The learning rate in the optimizer should be 0.001 as defined above.
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# PROBLEM 2.3 (15 Points) Complete the training process. The training process has 
	# two phases: a forward pass and backward propagation.
	# When you run the training algorithm, the loss will not decrease monotonically at 
	# each iteration, because we are using minibatches. The loss may go up and down, but 
	# it should generally decrease over time given the small learning rate.

	total_step = len(train_loader)
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			# Move tensors to the configured device
			images = images.view(-1, 28 * 28).to(device)
			labels = labels.to(device)

			# Forward pass
			# The forward process computes the loss of each iteration on each sample
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backward and using the optimizer to update the parameters
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

 
			# Below, an epoch corresponds to one pass through all of the samples.
			# Each training step corresponds to a parameter update using 
			# a gradient computed on a minibatch of 100 samples 
			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
					  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

	# Test the model
	# In the test phase, we don't need to compute gradients (for memory efficiency)
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in test_loader:
			images = images.reshape(-1, 28 * 28).to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

def parseCSV(file):
    data = pd.read_csv(file)
    return data

csv_data = parseCSV("language_dataset.csv").to_dict('split')['data']
category_lines = defaultdict(list)
for entry in csv_data:
    category_lines[entry[1]].append(entry[0])

all_categories = list(category_lines.keys())
n_categories = len(all_categories)

# print("-----Our Categories-----")
# print(all_categories)
# print('----category lines----')
# print(category_lines)
# print('----n_categories-----')
# print(n_categories)


n_categories = len(all_categories)

# -----------------------
# Turn names into tensors
# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
	return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor


# Turn a line into a <line_length x 1 x n_letters> tensor
def lineToTensor(line):
	tensor = torch.zeros(len(line), 1, n_letters)
	for li, letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor


def categoryFromOutput(output):
	top_n, top_i = output.topk(1)
	category_i = top_i[0].item()
	return all_categories[category_i], category_i


def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
	line_tensor = lineToTensor(line)
	return category, line, category_tensor, line_tensor

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		# Put the declaration of RNN network here
		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		# Put the computation for forward pass here
		combined = torch.cat((input, hidden), 1)
		output = self.i2o(combined)
		output = self.softmax(output)
		hidden = self.i2h(combined)

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

def train_iteration_CharRNN(learning_rate, category_tensor, line_tensor):
	criterion = nn.NLLLoss()
	hidden = rnn.initHidden()
	rnn.zero_grad()

	# The forward process
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	# The backward process
	loss = criterion(output, category_tensor)
	loss.backward()

	# Add parameters' gradients to their values, multiplied by learning rate
	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.item()


def train_charRNN(n_iters, learning_rate):
	print_every = 1000

	current_loss = 0

	def timeSince(since):
		now = time.time()
		s = now - since
		m = math.floor(s / 60)
		s -= m * 60
		return '%dm %ds' % (m, s)

	start = time.time()

	for iter in range(1, n_iters + 1):
		category, line, category_tensor, line_tensor = randomTrainingExample()
		output, loss = train_iteration_CharRNN(learning_rate, category_tensor, line_tensor)
		current_loss += loss

		# Print iter number, loss, name and guess
		if iter % print_every == 0:
			guess, guess_i = categoryFromOutput(output)
			correct = '✓' if guess == category else '✗ (%s)' % category
			print('%d %d%% (%s) %.4f %s / %s %s' % (
				iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
			print('Average loss: %.4f' % (current_loss/print_every))
			current_loss = 0

	torch.save(rnn, 'char-rnn-classification.pt')


# Finish the prediction function to provide predictions for any 
# input string (name) from the user
def predict(input_line, n_predictions=7):
	print("Prediction for %s:" % input_line)
	hidden = rnn.initHidden()

	# Generate the input for RNN
	line_tensor = lineToTensor(input_line)
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	# Get the value and index of top K predictions from the output
	# Then apply Softmax function on the scores of all category predictions so we can 
	# output the probabilities that this name belongs to different languages.
	topv, topi = output.data.topk(n_predictions, 1, True)
	softmax = nn.Softmax(dim=1)
	top_prob = softmax(topv)
	predictions = []
 
	for i in range(3):
		value = topv[0][i]
		prob = top_prob[0][i] * 100
		category_index = topi[0][i]
		print('%s Probability: (%.2f), Score: (%.2f)' % (all_categories[category_index], prob, value))
		predictions.append([value, all_categories[category_index]])
	return predictions

if __name__ == "__main__":
	print("Problem 2")
	train_and_test_simple_net(28 * 28, 200, 10)

	print("\nProblem 3")
	train_charRNN(15000, 0.005)

	csv_data = parseCSV("test_data.csv").to_dict('split')['data']
	test_words = defaultdict(list)
	for entry in csv_data:
		test_words[entry[1]].append(entry[0])


	print("Testing for English: ")
	for i in test_words["English"]: 
		predict(i)

	print("Testing for Spanish: ")
	for i in test_words["Spanish"]: 
		predict(i)

	print("Testing for German: ")
	for i in test_words["German"]: 
		predict(i)

	print("Testing for Italian: ")
	for i in test_words["Italian"]: 
		predict(i)

	print("Testing for French: ")
	for i in test_words["French"]: 
		predict(i)

	print("Testing for Primitive Elvish: ")
	for i in test_words["Primitive Elvish"]: 
		predict(i)

	print("Testing for Simlish: ")
	for i in test_words["Simlish"]: 
		predict(i)

	print("Testing for Other: ")
	for i in test_words["Other"]: 
	 	predict(i)