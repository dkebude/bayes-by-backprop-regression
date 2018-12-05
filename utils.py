import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def relu(x):
	return x*(x>0) #this is faster than max(x,0)

def relu_grad(x):
	return (x > 0).reshape(1,-1)
	
def gen_samples_toy(sample_count=250):
	std = 3
	x = np.random.uniform(-4,4,size=sample_count).reshape((-1,1)) # uniform sampling for x
	eps = np.random.normal(0,std**2,sample_count).reshape((-1,1)) # normal sampling for eps
	y = x**3+eps

	return (x,y), std # return a tuple of x's and y's

def gen_samples_paper(sample_count=250):
	std = np.sqrt(np.float64(0.02))
	x = np.random.uniform(-4,4,size=sample_count).reshape((-1,1)) # uniform sampling for x
	eps = np.random.normal(0,std**2,sample_count).reshape((-1,1)) # normal sampling for eps
	y = x+(0.3*np.sin(2*np.pi*(x+eps)))+(0.3*np.sin(4*np.pi*(x+eps)))+eps

	return (x,y), std # return a tuple of x's and y's

def gen_samples(sample_count=250, gen_paper_flag=False):
	if gen_paper_flag:
		return gen_samples_paper(sample_count)
	else:
		return gen_samples_toy(sample_count)

def gen_layer_sizes(num_layers, input_size, output_size, h_sizes):
#h_sizes must be a list, even if it will have a single element
	layer_sizes = []
	for i in range(num_layers):
		if i == 0: # input layer
			W = (input_size, h_sizes[i])
			b = (1,h_sizes[i])
		elif i == num_layers-1: # output layer
			W = (h_sizes[i-1], output_size)
			b = (1,output_size)
		else: # hidden layers
			W = (h_sizes[i-1], h_sizes[i])
			b = (1,h_sizes[i])
		layer_sizes.append((W,b))

	return layer_sizes

def init_distributions(layer_sizes):
# We will need a mu and a rho for each weight
	mus_W = []
	rhos_W = []
	mus_b = []
	rhos_b = []
	
	for W_shape, b_shape in layer_sizes:
		mu_W  = np.random.normal(0, 0.01, W_shape)
		rho_W = -5+np.random.uniform(-0.01, 0.01, W_shape)
		mu_b  = np.random.normal(0, 0.01, b_shape)
		rho_b = -5+np.random.uniform(-0.01, 0.01, b_shape)
		mus_W.append(mu_W)
		rhos_W.append(rho_W)
		mus_b.append(mu_b)
		rhos_b.append(rho_b)

	return mus_W, rhos_W, mus_b, rhos_b

def get_eps(layer_sizes):
# Sample eps from N(0,I) for each layer
	eps_W = []
	eps_b = []
	for W_shape, b_shape in layer_sizes:
		eps_W.append(np.random.normal(0,1,W_shape))  
		eps_b.append(np.random.normal(0,1,b_shape))  

	return eps_W, eps_b

def get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b):
# This one converts distributions into weights
	sigmas_W = [np.log(1. + np.exp(rho)) for rho in rhos_W]
	sigmas_b = [np.log(1. + np.exp(rho)) for rho in rhos_b]
	W = []
	b = []
	for i in range(len(mus_W)):
		W.append(mus_W[i]+np.multiply(sigmas_W[i],eps_W[i]))
		b.append(mus_b[i]+np.multiply(sigmas_b[i],eps_b[i]))
	return W, b, sigmas_W, sigmas_b #sigmas will later be used for posterior and gradient calculation 

def forward_pass(x, W, b, mus_W, mus_b, sigmas_W, sigmas_b, eps_W, eps_b):
# Go through the network
	z = []
	a = []
	# z and a will later be used for gradient calculation
	curr_input = x
	# Hidden layers
	for i in range(len(W)-1):
		mean_out = np.dot(curr_input, mus_W[i])+mus_b[i]
		std_out = np.sqrt(np.dot(np.square(curr_input),np.square(sigmas_W[i]))+sigmas_b[i])
		eps = np.random.normal(0,1,mean_out.shape)
		lin_out = mean_out+np.multiply(std_out,eps)
		z.append(lin_out) 
		curr_input = relu(lin_out)
		a.append(curr_input)
	# Output layers
	mean_out = np.dot(curr_input, mus_W[-1])+mus_b[-1]
	std_out = np.sqrt(np.dot(np.square(curr_input),np.square(sigmas_W[-1]))+sigmas_b[-1])
	eps = np.random.normal(0,1,mean_out.shape)
	y_hat = mean_out+np.multiply(std_out,eps)
	return y_hat, z, a, mean_out, std_out

def calc_log_likelihood(y_train, y_hat):
# Calculate log P(D|w) as log gaussian with unit variance
# -log P(D|w) = -log(1/sqrt(2*pi))-log(exp(-(y_train-y_hat)^2)/2)
# -log P(D|w) = -log(1)+0.5log(2*pi)+0.5(y_train-y_hat)^2
# -log P(D|w) = 0.5log(2*pi)+0.5(y_train-y_hat)^2
# Finally: -log P(D|w) = 0.5log(2*pi)+0.5(y_train-y_hat)^2
# Note that it is not a sum over all y_train samples
# We will calculate this loss at each iteration as we see (x_train,y_train) pairs
	return (0.5*np.square(y_train-y_hat))

def calc_log_prior(W, b, prior_std):
# Calculate log P(W) as log gaussian with variance
# log P(W) = log(1/(s*sqrt(2*pi)))-log(exp(-W^2/2s^2))
# log P(W) = -0.5log(2*pi)-log(s)-0.5(W^2)/(s^2)
	s = np.float64(prior_std)
	w_sum  = np.array([np.sum(-np.log(s)-0.5*np.square(w)/np.square(s)) for w in W])
	w_sum += np.array([np.sum(-np.log(s)-0.5*np.square(bias)/np.square(s)) for bias in b])
	return np.sum(w_sum)

def calc_log_var_posterior(W,b,sigmas_W,sigmas_b,eps_W,eps_b):
# Calculate log q(W|theta) as log gaussian N(W|mu,sigma^2)
# And we calculate sigmas as log(1+exp(rhos))
# sigmas = log(1+exp(rhos))
# q(W|theta) = log(1/(sigmas*sqrt(2*pi)))+log(exp(-(W-mus)^2/(2*sigmas^2)))
# q(W|theta) = log(1) - 0.5log(2*pi) - log(sigmas) -(W-mus)^2/(2*sigmas^2)
# q(W|theta) = -0.5log(2*pi)-log(sigmas) - 0.5(W-mus)^2/(sigmas^2)
# Note that W = mus + sigmas*eps
# q(W|theta) = -0.5log(2*pi)-log(sigmas) - 0.5(mus+sigmas*eps-mus)^2/(sigmas^2)
# q(W|theta) = -0.5log(2*pi)-log(sigmas) - 0.5(sigmas^2*eps^2)/(sigmas^2)
# Finally: q(W|theta) = -0.5log(2*pi)-log(sigmas) - 0.5(eps^2) -> This will help us with a lot of computational time
	log_var_posterior = 0
	for i in range(len(W)):
		log_var_posterior += np.sum(-np.log(sigmas_W[i])-0.5*np.square(eps_W[i]))
	for i in range(len(b)):
		log_var_posterior += np.sum(-np.log(sigmas_b[i])-0.5*np.square(eps_b[i]))
	return np.sum(log_var_posterior)

def calc_kl(W, b, sigmas_W, sigmas_b, eps_W, eps_b, sample_count, prior_std):
#KL divergence part of loss is simply log q(W|theta)-log p(W)
	return (1./sample_count)*(calc_log_var_posterior(W, b, sigmas_W, sigmas_b, eps_W, eps_b)-calc_log_prior(W,b, prior_std))

def variational_free_energy(W, b, sigmas_W, sigmas_b, eps_W, eps_b, y_train, y_hat, sample_count, prior_std):
	return np.sum(calc_kl(W, b, sigmas_W, sigmas_b, eps_W, eps_b, sample_count, prior_std)+calc_log_likelihood(y_train,y_hat))

# Explanations of the gradient calculator functions below are provided in the report

def rho_grad_kl(W, rhos, sigmas, eps, sample_count, prior_std):
	s = np.float64(prior_std)
	g_rhos = []
	for i, w in enumerate(W):
		g_rho = (1./sample_count)*((-1./(sigmas[i])+np.multiply(w/s**2,eps[i])))*(1./(1+np.exp(-rhos[i])))
		g_rhos.append(g_rho)
	return g_rhos

def mu_grad_kl(W, sample_count, prior_std):
	s = np.float64(prior_std)
	g_mus = [(1./sample_count)*w/s**2 for w in W]
	return g_mus

def W_grad_log_likelihood_2layer(W, b, z, a, y_train, y_hat, x_train):
# This is kind of a self-made autograd that can calculate gradients for any number of
# layers and parameter sizes, it only utilizes relu activation and log likelihood loss
# but it can be improved further
	gW = []
	gb = []
	gb_1 = (y_train-y_hat).reshape(-1,1)
	gw_1 = np.dot(a[0].T, gb_1)
	gb_0 = np.dot(gb_1, np.multiply(W[1].T,relu_grad(z[0])))
	gw_0 = np.dot(x_train, gb_0)
		
	gW.append(gw_0)
	gb.append(gb_0)
	gW.append(gw_1)
	gb.append(gb_1)
	return gW, gb

def W_grad_log_likelihood(W, b, z, a, y_train, y_hat, x_train,b_flag=False):
# This is kind of a self-made autograd that can calculate gradients for any number of
# layers and parameter sizes, it only utilizes relu activation and log likelihood loss
# but it can be improved further
	gW = []
	for i in range(len(W)):
		gw = (y_train-y_hat).reshape(-1,1)
		if not i == len(W):
			for j in reversed(list(set(range(len(W)))-set(range(i+1)))):
				gw = np.dot(gw,np.multiply(W[j].T,relu_grad(z[j-1])))
		if i == 0:
			x_train = x_train.reshape(-1,1)
			gw = np.dot(x_train.T,gw)
		else:
			gw = np.dot(a[i-1].T,gw)
		gW.append(gw)

	gB = []
	for i in range(len(b)):
		gb = (y_train-y_hat).reshape(-1,1)
		if not i == len(b):
			for j in reversed(list(set(range(len(b)))-set(range(i+1)))):
				gb = np.dot(gb,np.multiply(W[j].T,relu_grad(z[j-1])))
		gB.append(gb)	
	return gW, gB
		
def rho_grad_log_likelihood(gW, rhos, eps):
	g_rhos = []
	for i,gw in enumerate(gW):
		g_rho = -np.multiply(gw,eps[i]/(1+np.exp(-rhos[i])))
		g_rhos.append(g_rho)
	return g_rhos

def mu_grad_log_likelihood(gW):
	g_mus = list(map(lambda x: -x, gW))
	return g_mus

def mu_grad(gW, W, sample_count, prior_std):
	g_mus_kl = mu_grad_kl(W, sample_count, prior_std)
	g_mus_log_likelihood = mu_grad_log_likelihood(gW)
	g_mus = [g_mus_kl[i] + g_mus_log_likelihood[i] for i in range(len(g_mus_kl))]
	return g_mus

def rho_grad(gW, W, rhos, sigmas, eps, sample_count, prior_std):
	g_rhos_kl = rho_grad_kl(W, rhos, sigmas, eps, sample_count, prior_std)
	g_rhos_log_likelihood = rho_grad_log_likelihood(gW, rhos, eps)
	g_rhos = [g_rhos_kl[i] + g_rhos_log_likelihood[i] for i in range(len(g_rhos_kl))]
	return g_rhos