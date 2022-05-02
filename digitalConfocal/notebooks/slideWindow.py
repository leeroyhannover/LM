# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 00:34:22 2022

@author: nickh
"""

# sliding window demo


# import the necessary packages
import imutils
import matplotlib.pyplot as plt
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    stdMap = np.zeros((image.shape[:2]))  # careful the image is 3 channels
    
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            tPatch = image[y:y + windowSize[1], x:x + windowSize[0], 0]  # taking only 1 channel from 3
            tempStdPatch = np.ones((tPatch.shape)) * np.std((tPatch))
            # print(tempStdPatch)
            stdMap[y:y + windowSize[1], x:x + windowSize[0]] = tempStdPatch + stdMap[y:y + windowSize[1], x:x + windowSize[0]] 
            # print(tempStd)
     
    np.save('stdMap.npy', stdMap)
    # return stdMap
            
import argparse
import time
import cv2
import numpy as np

# read in the image
image = np.load('F:/LM/digitalConfocal/notebooks/testIMG.npy')

image = np.stack((image,)*3, axis=-1)
# (winW, winH) = (128, 128)
(winW, winH) = (256, 128)
# (winW, winH) = (512, 512)
STEP = 32
# STEP = 256


# no loop for the visualization

# test = sliding_window(image, stepSize=STEP, windowSize=(winW, winH))
# print(test)
# np.save('stdMap.npy', test)


# slide window scanning for one test
for (x, y, window) in sliding_window(image, stepSize=STEP, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    # draw the window during the process
    clone = image.copy() 
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)  # third param indicates the colour
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
       # time.sleep(0.025)
    time.sleep(0.025)

test1 = np.load('stdMap.npy')
plt.figure()
plt.imshow(test1, cmap='gray')

