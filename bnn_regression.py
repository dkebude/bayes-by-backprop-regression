import sys
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import *

def main(epochs, num_samples, seed, learning_rate, h_sizes, stop_flag, gen_paper_flag):
	np.random.seed(seed)
	(x_set, y_set), prior_std = gen_samples(num_samples, gen_paper_flag)
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
			
			# Calculate weights and biases
			eps_W, eps_b = get_eps(layer_sizes)
			W, b, sigmas_W, sigmas_b = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
			
			# Get predictions
			y_hat, z, a, mean_out, std_out = forward_pass(x_train, W, b, mus_W, mus_b, sigmas_W, sigmas_b, eps_W, eps_b)
			
			# Calculate loss
			total_loss += variational_free_energy(W, b, sigmas_W, sigmas_b, eps_W, eps_b, y_train, y_hat, num_samples, prior_std)
			
			# Get gradients
			gW, gb = W_grad_log_likelihood(W, b, z, a, y_train, y_hat, x_train, num_samples)
			g_mus_W = mu_grad(gW, W, num_samples, prior_std)
			g_rhos_W = rho_grad(gW, W, rhos_W, sigmas_W, eps_W, num_samples, prior_std)
			g_mus_b = mu_grad(gb, b, num_samples, prior_std)
			g_rhos_b = rho_grad(gb, b, rhos_b, sigmas_b, eps_b, num_samples, prior_std)
		
			# Parameter updates
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

	#Plotting starts here
	W, b, _,_ = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
	x_range = np.arange(-7.5,7.5,0.01)
	mean_out = np.array([forward_pass(x, W, b, mus_W, mus_b, sigmas_W, sigmas_b, eps_W, eps_b)[3][0] for x in x_range]).squeeze()
	std_out = np.array([forward_pass(x, W, b, mus_W, mus_b, sigmas_W, sigmas_b, eps_W, eps_b)[4][0] for x in x_range]).squeeze()
	if gen_paper_flag:
		y_truth = x_range+(0.3*np.sin(2*np.pi*(x_range)))+(0.3*np.sin(4*np.pi*(x_range)))
	else:
		y_truth = x_range**3
	plt.plot(x_set, y_set, 'kx', label='Data samples')
	plt.plot(x_range, y_truth, 'b', label='y_truth')
	plt.plot(x_range, mean_out, 'r', label='Mean of Regression Fit')
	plt.fill_between(x_range, mean_out-3*50*std_out, mean_out+3*50*std_out, color='r', alpha=0.4, label='+/- 3*Std. Dev. (Scaled by 50)')
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
	parser.add_argument('-z', '--hidden-size', default=[100], nargs='+', type=int, help='Hidden layer sizes')
	parser.add_argument('-p', '--stop-early', action='store_true', help='Activate early stopping')
	parser.add_argument('-g', '--gen-paper', action='store_true', help='Generate samples from the function from the paper')
	args = parser.parse_args()
	o = vars(args)
	gen_paper_flag = o['gen_paper']
	stop_flag = o['stop_early']
	epochs = o['epochs']
	num_samples = o['num_samples']
	seed = o['seed']
	lr = o['learning_rate']
	h_sizes = o['hidden_size']

	main(epochs, num_samples, seed, lr, h_sizes, stop_flag, gen_paper_flag)