# SDBSCAN
This repository contains the algorithm implementation in our work [A novel density-based clustering algorithm using nearest neighbor graph](https://www.sciencedirect.com/science/article/pii/S0031320320300121?via%3Dihub).
## How to use
The SDBSCAN package inherits from sklearn classes, and thus drops in neatly next to other sklearn cluster algorithms with an identical calling API. 

```
from sdbscan import SDBSCAN
from sklearn.datasets import make_blobs

data, _ = make_blobs(1000, centers=2)

labels = SDBSCAN(min_samples=20,
                 noise_percent=0.05
                 ).fit_predict(data)
```

## Additional functionality
This repository also contains the implementation of a fast *k*-NN graph construction algorithm based on [annoy libary](https://github.com/spotify/annoy) in "fastKNN.py". While it is more efficient than the kneighbors_graph() function in sklearn, the accuracy of the constructed *k*-NN graph is reduced, which may affect the performance of clustering.
You can change the *k*-NN graph construct algorithm in sdbscan() function which located in sdbscan.py.

## Dependencies
To use SDBSCAN, you should install the dependencies below:

- [scikit-learn](https://scikit-learn.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Annoy](https://github.com/spotify/annoy) (if using fastKNN function)

## Citing

To reference the SDBSCAN algorithm developed in this library please cite our paper published in Pattern Recognition.

```
@article{LI2020107206,
title = "A novel density-based clustering algorithm using nearest neighbor graph",
journal = "Pattern Recognition",
volume = "102",
pages = "107206",
year = "2020",
issn = "0031-3203",
doi = "https://doi.org/10.1016/j.patcog.2020.107206",
url = "http://www.sciencedirect.com/science/article/pii/S0031320320300121",
author = "Hao Li and Xiaojie Liu and Tao Li and Rundong Gan",
}
```
