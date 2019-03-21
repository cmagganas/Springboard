#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using Clustering
# ***
# This mini-project is based on [this blog post](http://blog.yhat.com/posts/customer-segmentation-using-python.html) by yhat. Please feel free to refer to the post for additional information, and solutions.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("poster")


# ## Data
# 
# The dataset contains information on marketing newsletters/e-mail campaigns (e-mail offers sent to customers) and transaction level data from customers. The transactional data shows which offer customers responded to, and what the customer ended up buying. The data is presented as an Excel workbook containing two worksheets. Each worksheet contains a different dataset.

# In[2]:


df_offers = pd.read_excel("C:/Users/acer.DESKTOP-NM2UJ22/Springboard/Springboard/clustering/WineKMC.xlsx", sheetname=0)
df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
df_offers.head()


# We see that the first dataset contains information about each offer such as the month it is in effect and several attributes about the wine that the offer refers to: the variety, minimum quantity, discount, country of origin and whether or not it is past peak. The second dataset in the second worksheet contains transactional data -- which offer each customer responded to.

# In[3]:


df_transactions = pd.read_excel("C:/Users/acer.DESKTOP-NM2UJ22/Springboard/Springboard/clustering/WineKMC.xlsx", sheetname=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1
df_transactions.head()


# ## Data wrangling

# We're trying to learn more about how our customers behave, so we can use their behavior (whether or not they purchased something based on an offer) as a way to group similar minded customers together. We can then study those groups to look for patterns and trends which can help us formulate future offers.
# 
# The first thing we need is a way to compare customers. To do this, we're going to create a matrix that contains each customer and a 0/1 indicator for whether or not they responded to a given offer. 

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> Create a data frame where each row has the following columns (Use the pandas [`merge`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html) and [`pivot_table`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html) functions for this purpose):
# <ul>
# <li> customer_name
# <li> One column for each offer, with a 1 if the customer responded to the offer
# </ul>
# <p>Make sure you also deal with any weird values such as `NaN`. Read the documentation to develop your solution.</p>
# </div>

# In[4]:


#your turn
new_df = pd.merge(df_offers, df_transactions)
final_df = new_df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n')
final_df = final_df.fillna(0).reset_index()
final_df.head()


# ## K-Means Clustering
# 
# Recall that in K-Means Clustering we want to *maximize* the distance between centroids and *minimize* the distance between data points and the respective centroid for the cluster they are in. True evaluation for unsupervised learning would require labeled data; however, we can use a variety of intuitive metrics to try to pick the number of clusters K. We will introduce two methods: the Elbow method, the Silhouette method and the gap statistic.

# ### Choosing K: The Elbow Sum-of-Squares Method
# 
# The first method looks at the sum-of-squares error in each cluster against $K$. We compute the distance from each data point to the center of the cluster (centroid) to which the data point was assigned. 
# 
# $$SS = \sum_k \sum_{x_i \in C_k} \sum_{x_j \in C_k} \left( x_i - x_j \right)^2 = \sum_k \sum_{x_i \in C_k} \left( x_i - \mu_k \right)^2$$
# 
# where $x_i$ is a point, $C_k$ represents cluster $k$ and $\mu_k$ is the centroid for cluster $k$. We can plot SS vs. $K$ and choose the *elbow point* in the plot as the best value for $K$. The elbow point is the point at which the plot starts descending much more slowly. 

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b></p> 
# <ul>
# <li> What values of $SS$ do you believe represent better clusterings? Why?
# <li> Create a numpy matrix `x_cols` with only the columns representing the offers (i.e. the 0/1 colums) 
# <li> Write code that applies the [`KMeans`](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering method from scikit-learn to this matrix. 
# <li> Construct a plot showing $SS$ for each $K$ and pick $K$ using this plot. For simplicity, test $2 \le K \le 10$.
# <li> Make a bar chart showing the number of points in each cluster for k-means under the best $K$.
# <li> What challenges did you experience using the Elbow method to pick $K$?
# </ul>
# </div>

# <div class="span5 alert alert-info">
# <p><b>Answers:</b></p> 
# <ul>
# <li> It is not at all clear which K will represent the best clustering. K= 3, 5 or 9 could have been chosen
# <li> The elbow method is unreliable even when averaging the scores over a multiple iterations of KMeans
# </ul>
# </div>

# In[5]:


# x_cols, cluster
x_cols = final_df.iloc[:,1:-1]
from sklearn.cluster import KMeans
print('K - KMean score')
scores = []
for K in range(2,11,1):
    k_means_temp = KMeans(n_clusters=K)
    temp_score=0
    for i in range(10):
        k_means_temp.fit(x_cols)
        temp_score += k_means_temp.inertia_
    temp_score=temp_score/10
    print('{0} - {1}'.format(K,temp_score))
    scores.append(temp_score)


# In[6]:


# K vs SS score plot
plt.plot(range(2,11), scores)
plt.xlabel('K')
plt.ylabel('SS')
plt.title('SS vs K Clusters')


# In[7]:


import numpy as np
score_change=np.array(scores)[:-1]-scores[1:]
score_change


# In[8]:


#calculate number of points per cluster in n_clusters=3
counts = np.bincount(KMeans(3).fit(x_cols).labels_)
pd.DataFrame(counts).plot(kind='bar')
plt.xticks([0,1,2], ['1','2','3'])
plt.xlabel('Cluster Number')
plt.ylabel('Frequency')
plt.title('Points per Cluster') # chart changes everytime


# ### Choosing K: The Silhouette Method
# 
# There exists another method that measures how well each datapoint $x_i$ "fits" its assigned cluster *and also* how poorly it fits into other clusters. This is a different way of looking at the same objective. Denote $a_{x_i}$ as the *average* distance from $x_i$ to all other points within its own cluster $k$. The lower the value, the better. On the other hand $b_{x_i}$ is the minimum average distance from $x_i$ to points in a different cluster, minimized over clusters. That is, compute separately for each cluster the average distance from $x_i$ to the points within that cluster, and then take the minimum. The silhouette $s(x_i)$ is defined as
# 
# $$s(x_i) = \frac{b_{x_i} - a_{x_i}}{\max{\left( a_{x_i}, b_{x_i}\right)}}$$
# 
# The silhouette score is computed on *every datapoint in every cluster*. The silhouette score ranges from -1 (a poor clustering) to +1 (a very dense clustering) with 0 denoting the situation where clusters overlap. Some criteria for the silhouette coefficient is provided in the table below.

# <pre>
# 
# | Range       | Interpretation                                |
# |-------------|-----------------------------------------------|
# | 0.71 - 1.0  | A strong structure has been found.            |
# | 0.51 - 0.7  | A reasonable structure has been found.        |
# | 0.26 - 0.5  | The structure is weak and could be artificial.|
# | < 0.25      | No substantial structure has been found.      |
# 
# </pre>
# Source: http://www.stat.berkeley.edu/~spector/s133/Clus.html

# Fortunately, scikit-learn provides a function to compute this for us (phew!) called [`sklearn.metrics.silhouette_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html). Take a look at [this article](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html) on picking $K$ in scikit-learn, as it will help you in the next exercise set.

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set III</h3>
# 
# <p><b>Exercise:</b> Using the documentation for the `silhouette_score` function above, construct a series of silhouette plots like the ones in the article linked above.</p>
# 
# <p><b>Exercise:</b> Compute the average silhouette score for each $K$ and plot it. What $K$ does the plot suggest we should choose? Does it differ from what we found using the Elbow method?</p>
# </div>

# In[9]:


# Your turn.
from __future__ import print_function
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
for n_clusters in range(2,11):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0,len(x_cols)+(n_clusters+1)*10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(x_cols)
    silhouette_avg = silhouette_score(x_cols, cluster_labels)
    print("n_clusters:", n_clusters, "\tAverage silhouette score:", silhouette_avg)
    sample_silhouette_values = silhouette_samples(x_cols, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values=sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i=ith_cluster_silhouette_values.shape[0]
        y_upper=y_lower+size_cluster_i
        color=cm.Spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,
                          facecolor=color,edgecolor=color,alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("Silhouette plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.Spectral(cluster_labels.astype(float) / n_clusters)
    pca = PCA(n_components=2).fit(x_cols)
    data2D = pca.transform(x_cols)
    ax2.scatter(data2D[:, 0], data2D[:, 1], marker='.', s=30, lw=0, alpha=1,
                c=colors, edgecolor='k')
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    ax2.set_title("Visualization of clustered data")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()


# In[10]:


scores2 = []
for value in range(2,11):
    kmeans2 = KMeans(n_clusters=value, random_state=10)
    kmeans2.fit(x_cols)
    labels = kmeans2.predict(x_cols)
    scores2.append(silhouette_score(x_cols, labels))


# In[11]:


index= np.arange(len(scores2))
plt.bar(index, scores2)
plt.xticks([0,1,2,3,4,5,6,7,8,9],['2','3','4','5','6','7','8','9','10'])
plt.xlabel('Number of clusters')
plt.ylabel('Ave. silhouette score')
plt.title('Silhouette score per number of clusters')


# <div class="span5 alert alert-info">
# This method is clearer and we can see that 5 clusters has the highest silhouette score on average

# ### Choosing $K$: The Gap Statistic
# 
# There is one last method worth covering for picking $K$, the so-called Gap statistic. The computation for the gap statistic builds on the sum-of-squares established in the Elbow method discussion, and compares it to the sum-of-squares of a "null distribution," that is, a random set of points with no clustering. The estimate for the optimal number of clusters $K$ is the value for which $\log{SS}$ falls the farthest below that of the reference distribution:
# 
# $$G_k = E_n^*\{\log SS_k\} - \log SS_k$$
# 
# In other words a good clustering yields a much larger difference between the reference distribution and the clustered data. The reference distribution is a Monte Carlo (randomization) procedure that constructs $B$ random distributions of points within the bounding box (limits) of the original data and then applies K-means to this synthetic distribution of data points.. $E_n^*\{\log SS_k\}$ is just the average $SS_k$ over all $B$ replicates. We then compute the standard deviation $\sigma_{SS}$ of the values of $SS_k$ computed from the $B$ replicates of the reference distribution and compute
# 
# $$s_k = \sqrt{1+1/B}\sigma_{SS}$$
# 
# Finally, we choose $K=k$ such that $G_k \geq G_{k+1} - s_{k+1}$.

# ### Aside: Choosing $K$ when we Have Labels
# 
# Unsupervised learning expects that we do not have the labels. In some situations, we may wish to cluster data that is labeled. Computing the optimal number of clusters is much easier if we have access to labels. There are several methods available. We will not go into the math or details since it is rare to have access to the labels, but we provide the names and references of these measures.
# 
# * Adjusted Rand Index
# * Mutual Information
# * V-Measure
# * Fowlkes–Mallows index
# 
# See [this article](http://scikit-learn.org/stable/modules/clustering.html) for more information about these metrics.

# ## Visualizing Clusters using PCA
# 
# How do we visualize clusters? If we only had two features, we could likely plot the data as is. But we have 100 data points each containing 32 features (dimensions). Principal Component Analysis (PCA) will help us reduce the dimensionality of our data from 32 to something lower. For a visualization on the coordinate plane, we will use 2 dimensions. In this exercise, we're going to use it to transform our multi-dimensional dataset into a 2 dimensional dataset.
# 
# This is only one use of PCA for dimension reduction. We can also use PCA when we want to perform regression but we have a set of highly correlated variables. PCA untangles these correlations into a smaller number of features/predictors all of which are orthogonal (not correlated). PCA is also used to reduce a large set of variables into a much smaller one.

# <div class="span5 alert alert-info">
# <h3>Checkup Exercise Set IV</h3>
# 
# <p><b>Exercise:</b> Use PCA to plot your clusters:</p>
# 
# <ul>
# <li> Use scikit-learn's [`PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) function to reduce the dimensionality of your clustering data to 2 components
# <li> Create a data frame with the following fields:
#   <ul>
#   <li> customer name
#   <li> cluster id the customer belongs to
#   <li> the two PCA components (label them `x` and `y`)
#   </ul>
# <li> Plot a scatterplot of the `x` vs `y` columns
# <li> Color-code points differently based on cluster ID
# <li> How do the clusters look? 
# <li> Based on what you see, what seems to be the best value for $K$? Moreover, which method of choosing $K$ seems to have produced the optimal result visually?
# </ul>
# 
# <p><b>Exercise:</b> Now look at both the original raw data about the offers and transactions and look at the fitted clusters. Tell a story about the clusters in context of the original data. For example, do the clusters correspond to wine variants or something else interesting?</p>
# </div>

# In[12]:


#your turn
from sklearn.decomposition import PCA
new_km = KMeans(n_clusters=5)
new_km.fit_transform(x_cols)
new_labels = new_km.predict(x_cols)
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(x_cols)
new_df = pd.DataFrame({'customer_name':final_df.customer_name, 'cluster_id':new_labels,
                       'x':data_reduced[:, 0], 'y':data_reduced[:, 1]})
plt.scatter(new_df.x, new_df.y, c=new_df.cluster_id)
plt.title('K = 5 after PCA')
plt.show()


# What we've done is we've taken those columns of 0/1 indicator variables, and we've transformed them into a 2-D dataset. We took one column and arbitrarily called it `x` and then called the other `y`. Now we can throw each point into a scatterplot. We color coded each point based on it's cluster so it's easier to see them.

# <div class="span5 alert alert-info">
# <h3>Exercise Set V</h3>
# 
# <p>As we saw earlier, PCA has a lot of other uses. Since we wanted to visualize our data in 2 dimensions, restricted the number of dimensions to 2 in PCA. But what is the true optimal number of dimensions?</p>
# 
# <p><b>Exercise:</b> Using a new PCA object shown in the next cell, plot the `explained_variance_` field and look for the elbow point, the point where the curve's rate of descent seems to slow sharply. This value is one possible value for the optimal number of dimensions. What is it?</p>
# </div>

# In[13]:


#your turn
# Initialize a new PCA model with a default number of components.
import sklearn.decomposition
pca = sklearn.decomposition.PCA()
pca.fit(x_cols)
variance = pca.explained_variance_
index2= np.arange(len(variance))
plt.bar(index2, variance)
plt.xlabel('dimension number')
plt.ylabel('explained variance')
plt.title('PCA dimension variance')


# <div class="span5 alert alert-info">
# Variance significantly slows decrease at dimension 3

# ## Other Clustering Algorithms
# 
# k-means is only one of a ton of clustering algorithms. Below is a brief description of several clustering algorithms, and the table provides references to the other clustering algorithms in scikit-learn. 
# 
# * **Affinity Propagation** does not require the number of clusters $K$ to be known in advance! AP uses a "message passing" paradigm to cluster points based on their similarity. 
# 
# * **Spectral Clustering** uses the eigenvalues of a similarity matrix to reduce the dimensionality of the data before clustering in a lower dimensional space. This is tangentially similar to what we did to visualize k-means clusters using PCA. The number of clusters must be known a priori.
# 
# * **Ward's Method** applies to hierarchical clustering. Hierarchical clustering algorithms take a set of data and successively divide the observations into more and more clusters at each layer of the hierarchy. Ward's method is used to determine when two clusters in the hierarchy should be combined into one. It is basically an extension of hierarchical clustering. Hierarchical clustering is *divisive*, that is, all observations are part of the same cluster at first, and at each successive iteration, the clusters are made smaller and smaller. With hierarchical clustering, a hierarchy is constructed, and there is not really the concept of "number of clusters." The number of clusters simply determines how low or how high in the hierarchy we reference and can be determined empirically or by looking at the [dendogram](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.cluster.hierarchy.dendrogram.html).
# 
# * **Agglomerative Clustering** is similar to hierarchical clustering but but is not divisive, it is *agglomerative*. That is, every observation is placed into its own cluster and at each iteration or level or the hierarchy, observations are merged into fewer and fewer clusters until convergence. Similar to hierarchical clustering, the constructed hierarchy contains all possible numbers of clusters and it is up to the analyst to pick the number by reviewing statistics or the dendogram.
# 
# * **DBSCAN** is based on point density rather than distance. It groups together points with many nearby neighbors. DBSCAN is one of the most cited algorithms in the literature. It does not require knowing the number of clusters a priori, but does require specifying the neighborhood size.

# ### Clustering Algorithms in Scikit-learn
# <table border="1">
# <colgroup>
# <col width="15%" />
# <col width="16%" />
# <col width="20%" />
# <col width="27%" />
# <col width="22%" />
# </colgroup>
# <thead valign="bottom">
# <tr><th>Method name</th>
# <th>Parameters</th>
# <th>Scalability</th>
# <th>Use Case</th>
# <th>Geometry (metric used)</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr><td>K-Means</span></a></td>
# <td>number of clusters</td>
# <td>Very large<span class="pre">n_samples</span>, medium <span class="pre">n_clusters</span> with
# MiniBatch code</td>
# <td>General-purpose, even cluster size, flat geometry, not too many clusters</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Affinity propagation</td>
# <td>damping, sample preference</td>
# <td>Not scalable with n_samples</td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr><td>Mean-shift</td>
# <td>bandwidth</td>
# <td>Not scalable with <span class="pre">n_samples</span></td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Spectral clustering</td>
# <td>number of clusters</td>
# <td>Medium <span class="pre">n_samples</span>, small <span class="pre">n_clusters</span></td>
# <td>Few clusters, even cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr><td>Ward hierarchical clustering</td>
# <td>number of clusters</td>
# <td>Large <span class="pre">n_samples</span> and <span class="pre">n_clusters</span></td>
# <td>Many clusters, possibly connectivity constraints</td>
# <td>Distances between points</td>
# </tr>
# <tr><td>Agglomerative clustering</td>
# <td>number of clusters, linkage type, distance</td>
# <td>Large <span class="pre">n_samples</span> and <span class="pre">n_clusters</span></td>
# <td>Many clusters, possibly connectivity constraints, non Euclidean
# distances</td>
# <td>Any pairwise distance</td>
# </tr>
# <tr><td>DBSCAN</td>
# <td>neighborhood size</td>
# <td>Very large <span class="pre">n_samples</span>, medium <span class="pre">n_clusters</span></td>
# <td>Non-flat geometry, uneven cluster sizes</td>
# <td>Distances between nearest points</td>
# </tr>
# <tr><td>Gaussian mixtures</td>
# <td>many</td>
# <td>Not scalable</td>
# <td>Flat geometry, good for density estimation</td>
# <td>Mahalanobis distances to  centers</td>
# </tr>
# <tr><td>Birch</td>
# <td>branching factor, threshold, optional global clusterer.</td>
# <td>Large <span class="pre">n_clusters</span> and <span class="pre">n_samples</span></td>
# <td>Large dataset, outlier removal, data reduction.</td>
# <td>Euclidean distance between points</td>
# </tr>
# </tbody>
# </table>
# Source: http://scikit-learn.org/stable/modules/clustering.html

# <div class="span5 alert alert-info">
# <h3>Exercise Set VI</h3>
# 
# <p><b>Exercise:</b> Try clustering using the following algorithms. </p>
# <ol>
# <li>Affinity propagation
# <li>Spectral clustering
# <li>Agglomerative clustering
# <li>DBSCAN
# </ol>
# <p>How do their results compare? Which performs the best? Tell a story why you think it performs the best.</p>
# </div>
# 

# In[14]:


# calculate silhouette_score for AffinityPropagation
from sklearn.cluster import AffinityPropagation
ap_labels = AffinityPropagation().fit_predict(x_cols)
silhouette_score(x_cols, ap_labels)


# In[15]:


# calculate silhouette_score for SpectralClustering
from sklearn.cluster import SpectralClustering
sc_labels = SpectralClustering(n_clusters=5).fit_predict(x_cols)
silhouette_score(x_cols, sc_labels)


# In[16]:


# calculate silhouette_score for AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
ac_labels = AgglomerativeClustering(n_clusters=5).fit_predict(x_cols)
silhouette_score(x_cols, ac_labels)


# In[17]:


# calculate silhouette_score for DBSCAN
from sklearn.cluster import DBSCAN
db_labels = DBSCAN(eps=1).fit_predict(x_cols) # multiple tests with the sihouette_score showed eps=1 to be the best
silhouette_score(x_cols, db_labels)


# <div class="span5 alert alert-info">
# Agglomerative clustering has the higest silhouette score, even higher than Kmeans. Agglomerative clustering is more prone to overfitting, but the Kmeans clusters overlapped quite heavily, so Agglomerative clustering must have been able to to divide the clusters out better than kmeans and score higher with more clusters.
