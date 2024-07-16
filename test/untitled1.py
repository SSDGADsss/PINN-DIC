# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:48:32 2024

@author: 28622
"""

import cv2
import numpy as np

# Create two simple binary images
image1 = np.zeros((300, 300), dtype=np.uint8)
image2 = np.zeros((300, 300), dtype=np.uint8)

# Draw a white rectangle in image1
cv2.rectangle(image1, (50, 50), (250, 250), 255, -1)

# Draw a white circle in image2
cv2.circle(image2, (100, 150), 100, 255, -1)

# Perform the bitwise OR operation
# result = cv2.bitwise_or(image1, image1)
# self.roi = cv2.bitwise_and(self.roi, cv2.bitwise_not(self.new_roi))
result = cv2.bitwise_and(image1, cv2.bitwise_not(image2))

# Display the images
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Bitwise OR Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
