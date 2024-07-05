# if __name__ == "__main__":
#     print('hello world!')

import torch
import torchvision

torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms]))

s = 'lolgpsaT'
