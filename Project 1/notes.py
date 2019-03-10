#!/usr/bin/env python
# coding: utf-8

# # Machine Learning
# ## Miniproject 1
# Dragi Kamov and Aadil Kumar

# ## Used libraries

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt


# # Loading data

# In[2]:


vectors = np.loadtxt('data/mfeat-pix.txt')
zeros = vectors[0:200]


# ## Visualization functions

# In[3]:


def showImage(vectors):
    imageMatrix = vectors.reshape(16, 15)
    plt.imshow(imageMatrix, cmap = 'gray')
    plt.show()
def imageCat(vectors): 
    for rows in vectors: 
        showImage(rows)


# ## Implementation of k-means algorithm

# In[4]:


def distance(p1, p2):
    diff = p1 - p2
    s = np.sum(np.power(diff, 2))
    return np.sqrt(s)


# In[5]:


def classify(pixels, centers): 
    num_centers, _ = centers.shape
    num_pixels, _ = pixels.shape
    distances = np.zeros((num_centers, num_pixels))
    
    for i in range(num_centers): 
        for j in range(num_pixels): 
            distances[i][j] = distance(centers[i], pixels[j])
    return np.argmin(distances, axis = 0)


# In[6]:


def find_cluster(pixels, centers): 
    num_centers, _ = centers.shape
    num_pixels, dim = pixels.shape
    
    classified = classify(pixels, centers)
    for c in range(num_centers): 
        sum = np.zeros(dim)
        count = 0 
        for i in range(num_pixels): 
            if(classified[i] == c): 
                sum += pixels[i]
                count += 1
        mean = sum / count 
        centers[c] = mean
    return centers 


# In[7]:


def k_means(pixels, k, iterations): 
    centers = pixels[0:k]
    for i in range(iterations): 
        centers = find_cluster(pixels, centers)
    return centers


# ## k = 1

# In[8]:


centers = k_means(zeros, 1, 10)
imageCat(centers)


# ## k = 2

# In[9]:


centers = k_means(zeros, 2, 10)
imageCat(centers)


# ## k = 3

# In[10]:


centers = k_means(zeros, 3, 10)
imageCat(centers)


# ## k = 200

# In[11]:


centers = k_means(zeros, 200, 10)
imageCat(centers)

