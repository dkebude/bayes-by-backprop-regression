import sys
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import *

def main(epochs, num_samples, seed, learning_rate, h_sizes, stop_flag):
	np.random.seed(seed)
	(x_set, y_set) = gen_samples(num_samples)
	layer_sizes = gen_layer_sizes(len(h_sizes)+1, 1, 1, h_sizes)
	mus_W, rhos_W, mus_b, rhos_b = init_distributions(layer_sizes)
	
	prev_loss = 0
	epochs_list = []
	losses = []
	for e in range(epochs):
		print '-------------------------'
		print 'Epoch', e+1
		total_loss = 0
		for i, x_train in enumerate(x_set):
			y_train = y_set[i]
			
			eps_W, eps_b = get_eps(layer_sizes)
			W, b, sigmas_W, sigmas_b = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
			
			y_hat, z, a = forward_pass(x_train, W, b)
			
			total_loss += variational_free_energy(W, b, sigmas_W, sigmas_b, eps_W, eps_b, y_train, y_hat, num_samples)
			
			gW, gb = W_grad_log_likelihood(W, b, z, a, y_train, y_hat, x_train, num_samples)
			g_mus_W = mu_grad(gW, W, num_samples)
			g_rhos_W = rho_grad(gW, W, rhos_W, sigmas_W, eps_W, num_samples)
			g_mus_b = mu_grad(gb, b, num_samples)
			g_rhos_b = rho_grad(gb, b, rhos_b, sigmas_b, eps_b, num_samples)
		
			for i in range(len(mus_W)):
				mus_W[i]  = mus_W[i]  - lr*g_mus_W[i]
				rhos_W[i] = rhos_W[i] - lr*g_rhos_W[i]
				mus_b[i]  = mus_b[i]  - lr*g_mus_b[i]
				rhos_b[i] = rhos_b[i] - lr*g_rhos_b[i]
			
		curr_loss = total_loss/num_samples
		epochs_list.append(e)
		losses.append(curr_loss)
		print 'Loss:', curr_loss
		if stop_flag and (abs(curr_loss-prev_loss) <= learning_rate):
			break
		prev_loss = curr_loss

	epochs = np.array(epochs_list)
	losses = np.array(losses)
	plt.plot(epochs, losses, 'tab:orange', label='Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title("Loss vs. Epochs")
	plt.legend(loc='best', fancybox=True, shadow=True)
	plt.show()

	W, b, _,_ = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
	x_range = np.arange(-0.2,0.6,0.01)
	y_range = np.array([forward_pass(x, W, b)[0][0] for x in x_range])
	plt.plot(x_set, y_set, 'kx', label='Data samples')
	plt.plot(x_range, y_range, 'r', label='Regression Fit')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title("Fit over data")
	plt.legend(loc='best', fancybox=True, shadow=True)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Regression via Bayes by Backprop',
 										 prog='python bnn_regression.py')
	parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs to run')
	parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='Define learning rate')
	parser.add_argument('-n', '--num-samples', default=500, type=int, help='Number of (x,y) samples to generate for training')
	parser.add_argument('-s', '--seed', default=None, type=int, help='Define random seed')
	parser.add_argument('-z', '--hidden-size', default=[32, 64], nargs='+', type=int, help='Hidden layer sizes')
	parser.add_argument('-p', '--stop-early', action='store_true', help='Activate early stopping')
	args = parser.parse_args()
	o = vars(args)
	stop_flag = o['stop_early']
	epochs = o['epochs']
	num_samples = o['num_samples']
	seed = o['seed']
	lr = o['learning_rate']
	h_sizes = o['hidden_size']

	main(epochs, num_samples, seed, lr, h_sizes, stop_flag)