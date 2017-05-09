# HomeWork6-239-v2

The objectives of this repo are the following:

	Implement the K-Means algorithm
Deal with text data (news records) in document-term sparse matrix format.
Design a proximity function for text data
Think about the Curse of Dimensionality
Think about best metrics for evaluating clustering solutions.

NMI: 0.5177

####Steps:
1. Read files train.dat and make list linesOfTrainData
2. After that calculation for featureList is done: intersection of unique
words from linesOfTrainData and unique words from linesOfTestData
is considered for this.
3. Now we calculated the CSR matrix from both training_list
4. CSR matrix is converted to Dense vector for computing the k-means.
5. As we have 7 clusters, we initialize first 7 points as centroids.
6. For every item in the list, we can check the distance between the
centroid and that item.
a. Assign the cluster values for the item, who is nearest to the
centroid of that cluster.
7. After the assignment, Recompute the centers by averaging the
values of items in the inside each cluster.
8. Write the results to format.dat



