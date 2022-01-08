# Awesome-Optimizer
Collect optimizer related papers, data, repositories

| Title                                           |  Year    | Optimizer       | Published                                  | Code                                              | Keywords                                  |
| ---------------------- | ---------------------- | ---------|-------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------|
|Adaptive Subgradient Methods for Online Learning and Stochastic Optimization|2011|AdaGrad|[jmlr](https://jmlr.org/papers/v12/duchi11a.html)|[code](https://github.com/mlpack/ensmallen/tree/master/include/ensmallen_bits/ada_grad)|Gradient descent|
|ADADELTA: An Adaptive Learning Rate Method|2012|ADADELTA|[arxiv](https://arxiv.org/abs/1212.5701v1)|[code](https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adadelta.py\#L6)|Gradient descent|
|A Stochastic Gradient Method with an Exponential Convergence Rate for Finite Training Sets|2012|SAG|[arxiv](https://arxiv.org/abs/1202.6258)||variance reduced|
|Accelerating stochastic gradient descent using predictive variance reduction|2013|SVRG|[neurips](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)|[code](https://github.com/kilianFatras/variance_reduced_neural_networks)|variance reduced|
|Adam: A Method for Stochastic Optimization|2014|Adam|[arxiv](https://arxiv.org/abs/1412.6980)|[code](https://paperswithcode.com/paper/adam-a-method-for-stochastic-optimization)|Gradient descent|
|SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives|2014|SAGA|[arxiv](https://arxiv.org/abs/1407.0202)|[code](https://github.com/elmahdichayti/SAGA)|variance reduced|
|Adam: A Method for Stochastic Optimization|2015|AdaMax|[arxiv](https://arxiv.org/abs/1412.6980)|[code](https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adamax.py#L5)|Gradient descent|
|Scale-Free Algorithms for Online Linear Optimization|2015|AdaFTRL|[arxiv](https://arxiv.org/abs/1502.05744)||Gradient descent|
|AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks|2017|AdaBatch|[arxiv](https://arxiv.org/abs/1712.02029)|[code](https://github.com/GXU-GMU-MICCAI/AdaBatch-numerical-experiments)|Gradient descent|
|AdaComp : Adaptive Residual Gradient Compression for Data-Parallel Distributed Training|2017|AdaComp|[arxiv](https://arxiv.org/abs/1712.02679)||Gradient descent|
|SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient|2017|SARAH|[arxiv](https://arxiv.org/abs/1703.00102)||variance reduced|
|Adafactor: Adaptive Learning Rates with Sublinear Memory Cost|2018|Adafactor|[arxiv](https://arxiv.org/abs/1804.04235)|[code](https://github.com/DeadAt0m/adafactor-pytorch)|Gradient descent|
|Quasi-hyperbolic momentum and Adam for deep learning|2018|QHAdam|[arxiv](https://arxiv.org/abs/1810.06801)|[code](https://github.com/facebookresearch/qhoptim)|Gradient descent|
|Online Adaptive Methods, Universality and Acceleration|2018|AcceleGrad|[arxiv](https://arxiv.org/abs/1809.02864)||Gradient descent|
|Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods|2018|AdaBayes|[arxiv](https://arxiv.org/abs/1807.07540)|[code](https://github.com/LaurenceA/adabayes)|Gradient descent|
|On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization|2018|AdaFom|[arxiv](https://arxiv.org/abs/1808.02941)||Gradient descent|
|On the Convergence of Adam and Beyond|2019|AMSGrad|[arxiv](https://arxiv.org/abs/1904.09237)|[code](https://github.com/pytorch/pytorch/blob/b7bda236d18815052378c88081f64935427d7716/torch/optim/adam.py#L6)|Gradient descent|
|Local AdaAlter: Communication-Efficient Stochastic Gradient Descent with Adaptive Learning Rates|2019|AdaAlter|[arxiv](https://arxiv.org/abs/1911.09030)|[code](https://github.com/xcgoner/AISTATS2020-AdaAlter-GluonNLP)|Gradient descent|
|Adaptive Gradient Methods with Dynamic Bound of Learning Rate|2019|AdaBound|[arxiv](https://arxiv.org/abs/1902.09843)|[code](https://github.com/Luolc/AdaBound)|Gradient descent|
|Does Adam optimizer keep close to the optimal point?|2019|AdaFix|[arxiv](https://arxiv.org/abs/1911.00289)||Gradient descent|
|Adaloss: Adaptive Loss Function for Landmark Localization|2019|Adaloss|[arxiv](https://arxiv.org/abs/1908.01070)||Gradient descent|
|A new perspective in understanding of Adam-Type algorithms and beyond|2019|AdamAL|[openreview](https://openreview.net/forum\?id\=SyxM51BYPB)|[code](https://www.dropbox.com/s/qgqhg6znuimzci9/adamAL.py\?dl\=0)|Gradient descent|
|On the Convergence of Adam and Beyond|2019|AdamNC|[arxiv](https://arxiv.org/abs/1904.09237)||Gradient descent|
|AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients|2020|AdaBelief|[arxiv](https://arxiv.org/abs/2010.07468)|[code](https://github.com/juntang-zhuang/Adabelief-Optimizer)|Gradient descent|
|ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning|2020|ADAHESSIAN|[arxiv](https://arxiv.org/abs/2006.00719)|[code](https://github.com/amirgholami/adahessian)|Gradient descent|
|Adai: Separating the Effects of Adaptive Learning Rate and Momentum Inertia|2020|Adai|[arxiv](https://arxiv.org/abs/2006.15815)|[code](https://github.com/zeke-xie/adaptive-inertia-adai)|Gradient descent|
|Adam<sup>+</sup>: A Stochastic Method with Adaptive Variance Reduction|2020|Adam<sup>+</sup>|[arxiv](https://arxiv.org/abs/2011.11985)||Gradient descent|
|Adam with Bandit Sampling for Deep Learning|2020|Adambs|[arxiv](https://arxiv.org/abs/2010.12986)|[code](https://github.com/forestliurui/Adam-with-Bandit-Sampling)|Gradient descent|
|Dynamic Game Theoretic Neural Optimizer|2021|DGNOpt|[arxiv](https://arxiv.org/abs/2105.03788)||Gradient descent|
|Why are Adaptive Methods Good for Attention Models?|2020|ACClip|[arxiv](https://arxiv.org/abs/1912.03194)||Gradient descent|
