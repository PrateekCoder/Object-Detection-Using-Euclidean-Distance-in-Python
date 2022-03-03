# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:25:37 2022

@author: Prateek
"""
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import rescale
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.spatial.distance import cdist

#Reading Image
wolf = imread('wolf.png', as_gray= True)
wolf_sheep = imread('sheepwolf.png', as_gray=True)

#Show Image
plt.imshow(wolf)
plt.imshow(wolf_sheep)

#Rescaling
wolf_rescale = rescale(wolf, scale=(0.05, 0.05))
wolf_sheep_rescale = rescale(wolf_sheep, scale=(0.05, 0.05))

#Show Image After Rescaling
#plt.imshow(wolf_rescale)
#plt.imshow(wolf_sheep_rescale)

#Flattening
wolf_rescale_flat = wolf_rescale.ravel()
window_shape = (wolf_rescale.shape[0], wolf_rescale.shape[1])

#Creating Patches
patches = extract_patches_2d(wolf_sheep_rescale, window_shape)
print("There were total of", len(patches), "patches created.")

#Flattening The Patches.
patches_flat = patches.ravel()

#Finding the coordinates logic function:
def coordinate(patches, wolf_sheep_rescale):
    distance = []
    sum_distance = []
    for i in range(len(patches)):
        dist = cdist(patches[i], wolf_rescale, metric='euclidean' )
        distance.append(dist)
    
    for i in range(len(patches)):
        dist = sum(sum(distance[i]))
        sum_distance.append(dist)
        
    min_distance = min(sum_distance)
    index = sum_distance.index(min_distance)
    
    y = (index/(wolf_sheep_rescale.shape[0]-wolf_rescale.shape[0]))/0.05
    x = (index%((wolf_sheep_rescale.shape[1]-wolf_rescale.shape[1]+1)))/0.05
    
    coordinates = (x,y)
    return coordinates

print("The coordinate of the Wolf among the flock of Sheep is:", coordinate(patches, wolf_sheep_rescale))