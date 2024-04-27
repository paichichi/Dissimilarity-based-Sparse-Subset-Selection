# Dissimilarity-based-Sparse-Subset-Selection

This repository contains the python implementation of the paper: Dissimilarity-based-Sparse-Subset-Selection

The paper aims to find the data representatives given the pairwise dissimilarities between the source set X and target set Y in an unsupervised framework. The paper formulated this as an optimization problem as a row-sparsity regularized trace minimization problem based on simultaneous sparse recovery theory where the regularization parameter puts a trade-off between the number of representatives and the encoding cost of  Y via representatives. The proposed algorithm is based on convex programming. While CVX can be used to solve this, but CVX cannot scale well with increase in the problem size. To address this, The authors uses Alternating Direction Methods of Multipliers (ADMM) which has quadratic complexity. ADMM allows to parallelize the algorithm, hence further reducing the computational time.

To cite this paper
E. Elhamifar, G. Sapiro and S. S. Sastry, "Dissimilarity-Based Sparse Subset Selection," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 11, pp. 2182-2197, 1 Nov. 2016, doi: 10.1109/TPAMI.2015.2511748.
keywords: {Computational modeling;Yttrium;Data models;Approximation algorithms;Optimization;Clustering algorithms;Computers;Representatives;pairwise dissimilarities;simultaneous sparse recovery;encoding;convex programming;ADMM optimization;sampling;clustering;outliers;model identification;time-series data;video summarization;activity clustering;scene recognition},

