
# coding: utf-8

# # Machine Learning
# ## Miniproject 1
# Dragi Kamov and Aadil Kumar

# ## Used libraries

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt


# ## Loading data

# In[2]:


vectors = np.loadtxt('data/mfeat-pix.txt')
zeros = vectors[0:200]


# ## Visualization functions

# In[3]:


def showImage(vectors):
    imageMatrix = vectors.reshape(16, 15)
    plt.imshow(imageMatrix, cmap = 'gray')
    plt.show()
    
def multImage(vectors): 
    for rows in vectors: 
        showImage(rows)


# ## Implementation of K-means algorithm

# In[4]:


def euclideanDistance(p1, p2):
    diff = p1 - p2
    s = np.sum(np.power(diff, 2))
    return np.sqrt(s)


# #### Returns the index of the points with the lowest distance to the centroids

# In[5]:


def classify(pixels, centroids): 
    numcentroids, _ = centroids.shape
    numPixels, _ = pixels.shape
    distances = np.zeros((numcentroids, numPixels))
    
    for i in range(numcentroids): 
        for j in range(numPixels): 
            distances[i][j] = euclideanDistance(centroids[i], pixels[j])
            
    return np.argmin(distances, axis = 0)


# #### Returns the center points of the clusters

# In[6]:


def findCluster(pixels, centroids): 
    numcentroids, _ = centroids.shape
    numPixels, dim = pixels.shape
    classified = classify(pixels, centroids)
    
    for c in range(numcentroids): 
        sum = np.zeros(dim)
        count = 0 
        for i in range(numPixels): 
            if(classified[i] == c): 
                sum += pixels[i]
                count += 1
                
        mean = sum / count 
        centroids[c] = mean
        
    return centroids 


# In[7]:


def kMeans(pixels, k, iterations): 
    centroids = pixels[0:k]
    for i in range(iterations): 
        centroids = findCluster(pixels, centroids)
    return centroids


# ## k = 1

# In[8]:


centroids = kMeans(zeros, 1, 10)
multImage(centroids)

mean = np.mean(zeros, axis = 0)
for i in range(10): 
    centroid = kMeans(zeros, 1, i)
    print("Euclidean Distance:", euclideanDistance(mean, centroid))
    plt.plot(i, euclideanDistance(mean, centroid), 'rx')
    plt.ylabel('Euclidean Distance')
    plt.xlabel('Iterations')
    plt.show


# #### As you can see the distance of the mean to centroid is 0 which shows that for K = 1 the centroid is just the mean, even for multiple iterations.  

# ## k = 2

# In[9]:


centroids = kMeans(zeros, 2, 10)
multImage(centroids)
#mean = np.mean(zeros, axis = 0)
for i in range(10): 
    centroid = kMeans(zeros, 2, i)
    print("Euclidean Distance:", euclideanDistance(mean, centroid))
    plt.plot(i, euclideanDistance(mean, centroid), 'rx')
    plt.ylabel('Euclidean Distance')
    plt.xlabel('Iterations')
    plt.show


# ## k = 3

# In[10]:


centroids = kMeans(zeros, 3, 10)
multImage(centroids)
#mean = np.mean(zeros, axis = 0)
for i in range(10): 
    centroid = kMeans(zeros, 3, i)
    print("Euclidean Distance:", euclideanDistance(mean, centroid))
    plt.plot(i, euclideanDistance(mean, centroid), 'rx')
    plt.ylabel('Euclidean Distance')
    plt.xlabel('Iterations')
    plt.show


# ## k = 200

# In[11]:


centroids = kMeans(zeros, 200, 1)
multImage(centroids)
#mean = np.mean(zeros, axis = 0)
for i in range(10): 
    centroid = kMeans(zeros, 200, i)
    print("Euclidean Distance:", euclideanDistance(mean, centroid))
    plt.plot(i, euclideanDistance(mean, centroid), 'rx')
    plt.ylabel('Euclidean Distance')
    plt.xlabel('Iterations')
    plt.show

