
# coding: utf-8

# In[1]:

import numpy as np

import scipy as sp
from scipy import spatial

from collections import defaultdict
from random import uniform
from math import sqrt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score

vectorizer = CountVectorizer()

from collections import defaultdict
from math import sqrt
import random


# In[2]:

with open("train.dat", "r") as fh:
    linesOfTrainData = fh.readlines()
print("Line count in training :" ,len(linesOfTrainData))

with open("format.dat", "r") as fh:
    linesOfFormat = fh.readlines()
print("Line count in format :" ,len(linesOfFormat))


# In[3]:

#for testing
#linesOfTrainData = linesOfTrainData[:100]
#print("Line count in training :" ,len(linesOfTrainData))


# In[4]:

vectorizer = CountVectorizer(lowercase = True)


# In[5]:

training_list = []

for td in linesOfTrainData:
    training_list.append(td)


# In[6]:

features =  set()
def feature_selection(data):
    for rows in data:
        i=1
        for feat in  rows.split():
            if i%2 != 0 : 
                #print feat
                features.add(feat)
            i=i+1


# In[7]:

feature_selection(training_list)
print(len(features))


# In[8]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(norm='l2', vocabulary=list(features))


# In[9]:

training_matrix =  tf.fit_transform(training_list)
training_feature_names = tf.get_feature_names() 


# In[10]:

denseTrainMatrix = training_matrix.todense()


# In[11]:

print(denseTrainMatrix)


# In[12]:

def densify(x, n):
    """Convert a sparse vector to a dense one."""
    d = [0] * n
    for i, v in x:
        d[i] = v
    return d


# In[13]:

"""def dist(x, c):
    """"""Euclidean distance between sample x and cluster center c.
    Inputs: x, a sparse vector
            c, a dense vector
    """"""
    sqdist = 0.
    for i, v in x:
        sqdist += (v - c[i]) ** 2
    return sqrt(sqdist)
"""


# In[14]:

def denseToListConvertor(a):
    if isinstance(a, list):
        return a
    else:    
        alist = a.tolist()
        return alist


# In[15]:

def computeAvg(avgList):
    print "Count of avg list: "+str(len(avgList))
    sum= avgList[0]
    for listElement in avgList[1:]:
        i=0
        for item in listElement:
            sum[i] = sum[i]+item    
            i = i+1
        
    #print "sum : "+str(sum)
    
    avg = sum
    i=0
    for item in sum:
        avg[i] = item/len(avgList)
        i=i+1
   
    return avg


# In[16]:

def dist_test(aVector, bVector,_iter):

    a =  denseToListConvertor(aVector)
    b =  denseToListConvertor(bVector)
    
    #print "len a[0] : "+ str(len(a[0]))
    #print "len b[0] : " + str(len(b[0]))
    
    #print "a[0][0] : "+ str(a[0][0])
    #print "b[0][0] : " + str(b[0][0])
    
    #print '---'
    
    dimensions = len(a[0])
    #print dimensions
    
    _sum = 0
    for dimension in xrange(dimensions):

        #print(dimension)
        if _iter ==0:
            #print "a[0][dimension] : "+str(a[0][dimension])
            #print "b: "+str(b)
            
            difference_sq = (a[0][dimension] - b[0][dimension]) ** 2
        else:
            difference_sq = (a[0][dimension] - b[dimension]) ** 2
        #print "difference_sq : "+str(difference_sq)
        _sum += difference_sq
        
    #print "Distance: "+ str(_sum)
    #print "----"
    return sqrt(_sum)


# In[17]:

#Goal of this method is to get the new centers
def mean(xs,cluster, k):
    i=0  
    returnCenters = []
    XS = xs.tolist()
    while (k > i) :
        avgList=[]
        for value in XS:
            #print "XS.index(value) : "+str(XS.index(value))
            #print " K : "+str(i)+" and cluster[XS.index(value)] : "+str(cluster[XS.index(value)])
            if cluster[XS.index(value)] == i:
                avgList.append(value)
                
        
        returnCenters.append(computeAvg(avgList)) #ComputeAvg will give us the avg of all the nodes 
        
        i+=1
        
        #print "New centers"+str(returnCenters)
    return returnCenters
        
            
        


# In[18]:

def getRandomCenters(xs,k):
    randomCenters = []
    i=0
    while i<k:
        randomCenters.append(xs[i].tolist())
        i+=1
        
    return randomCenters
    
       
        


# In[ ]:

def kmeans(k, xs, l, n_iter=30):
    # Initialize from random points.
    centers = getRandomCenters(xs,k)
    
    #Initialize clusters to 
    cluster = [None] * len(xs)
    
    #Iterate for n_iter
    for _ in xrange(n_iter):
        print "Iteration : "+str(_)
        #print "Centers : "+str(centers)
        
        #for each row in list xs
        for i, x in enumerate(xs):          
            
            #list of all the center distances
            listCenterDistance = []
            for center in centers:
                listCenterDistance.append(dist_test(x.tolist(), center ,_))        
            
            #print "listCenterDistance : "+str(listCenterDistance)
            #print "min(listCenterDistance) : "+str(min(listCenterDistance))
            #print "listCenterDistance.index(min(listCenterDistance)) : "+str(listCenterDistance.index(min(listCenterDistance)))
            cluster[i] = listCenterDistance.index(min(listCenterDistance))
              
        #Adjust the centers        
        centers = mean(xs,cluster,k) 
       
        print "cluster : "+str(cluster) 
        #print "--centers-- : "+ str(len(centers))
        
    
    return cluster


# In[ ]:

import re
import sys

KValue= 7
print("usage: %s k docs..." % KValue)

k=KValue


vocab = features
xs = denseTrainMatrix


cluster_ind = kmeans(k, xs, len(vocab))

print "Final cluster : "+str(cluster_ind)


# In[ ]:


f = open('format.dat', 'w')
count = 0
for clusterValue in cluster_ind:
    #print(clusterValue)
    f.write(str(clusterValue+1)+'\n')
    count+=1
print("count : ",count)
print("--The End--")


# In[ ]:




# In[ ]:



