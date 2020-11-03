# VAR-Tensor

"High-dimensional vector autoregressive time series modeling via tensor decomposition" 
by Wang, Zheng, Lian and Li (2020). 

The package includes estimation functions and real datasets used in this paper.

The Julia codes are based on Julia 1.4.0. Users have to use the updated Julia (after 1.2.0).
The required packages include the most updated version of Distributions, TensorToolbox, LinearAlgebra, SparseArrays and DelimitedFiles.

The macroeconomic dataset is provided by Gary Koop, "Forecasting with Medium and Large Bayesian VARs", Journal of Applied Econometrics, Vol. 28, No. 2, 2013, pp. 177-203. All data are in the spreadsheet es09_1.xls, and 40 macroeconomic series are extracted to macro40.csv.

This package is maintained by Di Wang <diwang@conenct.hku.hk>.
