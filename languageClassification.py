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

def parseCSV():
    