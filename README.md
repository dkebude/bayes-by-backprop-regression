# Bayes-by-Backprop Regression

A Python implementation of bayes by backprop regression from scratch including gradient calculations. 

Bayes by Backprop (BbB) is introduced in 2015 article [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424).

### Implemented with:

* python 2.7.12
* numpy 1.14.5
* matplotlib 2.2.3
* tensorflow 1.12.0 (for gradient checking only)

### Files:
* _utils.py_ contains the utility functions necessary for (BbB) implementation
* _grad_check.py_ contains:
	* a calculation of gradients via eager execution of tensorflow for autograd usage
	* grad_checking via a comparison of implemented gradients vs. tf gradients
* _bnn_regression.py_ contains the actual implementation of BbB, 
	* check -h for usage (hyperparameter choices, early stopping etc.)