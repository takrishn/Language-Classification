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
import numpy as np
import unicodedata
import string
import pandas as pd
from collections import defaultdict

def parseCSV(file):
    data = pd.read_csv(file)
    return data

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

csv_data = parseCSV("language_dataset.csv").to_dict('split')['data']
category_lines = defaultdict(list)
for entry in csv_data:
    category_lines[entry[1]].append(entry[0])

all_categories = list(category_lines.keys())
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


# PROBLEM 3.1 (20 points)  First, we need to create a recurrent neural network in Torch. 
# At each time step of a RNN network, the input and the output of the hidden layer are 
# combined. Refer to the PyTorch Tutorial 
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
# for the RNN structure and finish defining the following RNN model.
# The number of RNN units depends on the length of the input string,
# each character in a name corresponds to a time step.
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
		output = self.softmax(self.i2o(combined))
		hidden = self.i2h(combined)

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


# PROBLEM 3.2 (15 points) Finish the training function of this RNN network.
# Here the terminology is a bit different than Problem 2. The minibatch size used here
# here is 1, i.e., samples are randomly selected one at a time, a (stochastic) 
# gradient is computed and parameters are updated. (as defined in train_charRNN). 
# This operation (i.e., gradient and parameter update based on one sample) corresponds
# to a single "iteration" in the terminology below.
# In the train_iteration_CharRNN, you need to complete the training process of each
# iteration.
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
			correct = 'O' if guess == category else 'X (%s)' % category
			print('%d %d%% (%s) %.4f %s / %s %s' % (
				iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
			print('Average loss: %.4f' % (current_loss/print_every))
			current_loss = 0

	torch.save(rnn, 'char-rnn-classification.pt')


# Problem 3.3 (15 points) Finish the prediction function to provide predictions for any 
# input string (name) from the user
def predict(input_line, n_predictions=7):
	print("Predition for %s:" % input_line)
	hidden = rnn.initHidden()

	# Generate the input for RNN
	line_tensor = lineToTensor(input_line)
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	# Get the value and index of top K predictions from the output
	# Then apply Softmax function on the scores of all category predictions so we can 
	# output the probabilities that this name belongs to different languages.
	topv, topi = output.topk(n_predictions, 1, True)
	softmax = nn.Softmax(dim=1)
	top_prob = softmax(topv)
	predictions = []
 
	for i in range(n_predictions):
		value = topv[0][i].item()
		prob = top_prob[0][i].item()
		category_index = topi[0][i].item()
		print('%s Probability: (%.2f), Score: (%.2f)' % (all_categories[category_index], prob, value))
		predictions.append([value, all_categories[category_index]])
	return predictions

if __name__ == '__main__':
    train_charRNN(15000, 0.005)
    predict("event")
