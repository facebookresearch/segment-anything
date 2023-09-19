



'''
This file is used to evaluate the performance of the model.
It should be run after the training is done.
It will compare the predicted segmentation with the ground truth segmentation. vea dice coefficient.
'''

# Lets import the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2



# Lets define the function that will calculate the dice coefficient
def dice_coefficient(prediction, ground_truth):
    '''
    This function will calculate the dice coefficient of the prediction and the ground truth.
    It will return the dice coefficient.
    '''
    # Lets calculate the dice coefficient
    dice_coefficient = (2 * np.sum(prediction * ground_truth)) / (np.sum(prediction) + np.sum(ground_truth))
    # Lets return the dice coefficient
    return dice_coefficient



if __name__ == '__main__':

    # Test the function
    # Lets create a prediction and a ground truth
    prediction = np.array([[0, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    ground_truth = np.array([[0, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    # Let's calculate the dice coefficient
    print(dice_coefficient(prediction, ground_truth))



