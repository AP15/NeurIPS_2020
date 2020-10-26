# Brief explanation to reproduce the experimental results reported in the paper "Exact Recovery of Mangled Clusters with Same-Cluster QUeries"
# accepted at NeurIPS 2020.

# Make the plots:
Ensure to download all the data file (npz).
Run TestExp script to reproduce the experiments. 
The script will load the data for the 4 experiments (d=2, 4, 6, 8) and make the plots in the paper.

# How to use the code:
RECUR is implemented in the class is "ecc". The constructor requires the number of clusters and the margin.

The method clusterMonitor perform the clustering. It requires as input: 
- the X data; 
- the same-cluster-queries oracle;
- (see the class below) and the true labels. 

The method returns: 
- the clustering of the data in the form of labels;
- the number of queries made until completion;
- the accuracy per round in the form of a numpy array;
- and the number of queries made up to round i in the form of a numpy array.

The class "oracle" implements the same-cluster-queries oracle. The constructor requires the true labels in the form of a pandas
dataframe. The main method are:

label(i): returns the label of point i by making at most k same-cluster-queries
scq(idx, idy): return True is idx and idy belong to the same cluster.
