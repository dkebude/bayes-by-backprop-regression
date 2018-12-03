from utils import *
import numpy as np
import tensorflow as tf

def grad_check_tf_wb(num_samples, h_sizes):
	tf.enable_eager_execution()
	(x_set, y_set) = gen_samples(num_samples)
	layer_sizes = gen_layer_sizes(len(h_sizes)+1, 1, 1, h_sizes)
	mus_W, rhos_W, mus_b, rhos_b = init_distributions(layer_sizes)
	eps_W, eps_b = get_eps(layer_sizes)
	W, b, sigmas_W, sigmas_b = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
	w0  = tf.get_variable('w0',  layer_sizes[0][0], initializer=tf.constant_initializer(W[0]), dtype=tf.float64)
	b0  = tf.get_variable('b0',  layer_sizes[0][1], initializer=tf.constant_initializer(b[0]), dtype=tf.float64)
	w1  = tf.get_variable('w1',  layer_sizes[1][0], initializer=tf.constant_initializer(W[1]), dtype=tf.float64)
	b1  = tf.get_variable('b1',  layer_sizes[1][1], initializer=tf.constant_initializer(b[1]), dtype=tf.float64)
	
	x = tf.constant(x_set[0].reshape(-1,1), dtype=tf.float64)
	y = tf.constant(y_set[0].reshape(-1,1), dtype=tf.float64)
	
	with tf.GradientTape() as tape:
		z0 = tf.matmul(x,w0)+b0
		a0 = tf.nn.relu(z0)
		y_hat = tf.matmul(a0,w1)+b1
	
		loss = -tf_calc_log_likelihood(y, y_hat)
		gw0, gb0, gw1, gb1 = tape.gradient(loss,[w0,b0,w1,b1])
		
	x  = x.numpy()
	y  = y.numpy()
	z0 = [z0.numpy()]
	a0 = [a0.numpy()]
	y_hat = y_hat.numpy()

	gW, gb = W_grad_log_likelihood(W, b, z0, a0, y, y_hat, x, num_samples)
	print np.linalg.norm(gW[0]-gw0)
	print np.linalg.norm(gW[1]-gw1)
	print np.linalg.norm(gb[0]-gb0)
	print np.linalg.norm(gb[1]-gb1)

def grad_check_tf(num_samples, h_sizes):
	tf.enable_eager_execution()
	(x_set, y_set) = gen_samples(num_samples)
	layer_sizes = gen_layer_sizes(len(h_sizes)+1, 1, 1, h_sizes)
	mus_W, rhos_W, mus_b, rhos_b = init_distributions(layer_sizes)
	eps_W, eps_b = get_eps(layer_sizes)
	W, b, sigmas_W, sigmas_b = get_params(mus_W, rhos_W, mus_b, rhos_b, eps_W, eps_b)
	mu_W_0  = tf.get_variable('mu_W_0',  layer_sizes[0][0], initializer=tf.constant_initializer(mus_W[0]) , dtype=tf.float64)
	rho_W_0 = tf.get_variable('rho_W_0', layer_sizes[0][0], initializer=tf.constant_initializer(rhos_W[0]), dtype=tf.float64)
	mu_b_0  = tf.get_variable('mu_b_0',  layer_sizes[0][1], initializer=tf.constant_initializer(mus_b[0]) , dtype=tf.float64)
	rho_b_0 = tf.get_variable('rho_b_0', layer_sizes[0][1], initializer=tf.constant_initializer(rhos_b[0]), dtype=tf.float64)
	mu_W_1  = tf.get_variable('mu_W_1',  layer_sizes[1][0], initializer=tf.constant_initializer(mus_W[1]) , dtype=tf.float64)
	rho_W_1 = tf.get_variable('rho_W_1', layer_sizes[1][0], initializer=tf.constant_initializer(rhos_W[1]), dtype=tf.float64)
	mu_b_1  = tf.get_variable('mu_b_1',  layer_sizes[1][1], initializer=tf.constant_initializer(mus_b[1]) , dtype=tf.float64)
	rho_b_1 = tf.get_variable('rho_b_1', layer_sizes[1][1], initializer=tf.constant_initializer(rhos_b[1]), dtype=tf.float64)

	eps_w0 = tf.constant(eps_W[0], dtype=tf.float64)
	eps_b0 = tf.constant(eps_b[0], dtype=tf.float64)
	eps_w1 = tf.constant(eps_W[1], dtype=tf.float64)
	eps_b1 = tf.constant(eps_b[1], dtype=tf.float64)
	x = tf.constant(x_set[0].reshape(-1,1), dtype=tf.float64)
	y = tf.constant(y_set[0].reshape(-1,1), dtype=tf.float64)
	
	with tf.GradientTape() as tape:
		sigma_w0 = tf.nn.softplus(rho_W_0)
		sigma_b0 = tf.nn.softplus(rho_b_0)
		sigma_w1 = tf.nn.softplus(rho_W_1)
		sigma_b1 = tf.nn.softplus(rho_b_1)
		w0 = mu_W_0 + sigma_w0*eps_w0
		b0 = mu_b_0 + sigma_b0*eps_b0
		w1 = mu_W_1 + sigma_w1*eps_w1
		b1 = mu_b_1 + sigma_b1*eps_b1

		z0 = tf.matmul(x,w0)+b0
		a0 = tf.nn.relu(z0)
		y_hat = tf.matmul(a0,w1)+b1
	
		tf_loss = tf_variational_free_energy(w0,b0,w1,b1,sigma_w0,sigma_b0,sigma_w1,sigma_b1,eps_w0,eps_b0,eps_w1,eps_b1, y, y_hat, num_samples)
		g_mu_W_0, g_rho_W_0, g_mu_b_0, g_rho_b_0, g_mu_W_1, g_rho_W_1, g_mu_b_1, g_rho_b_1 = tape.gradient(tf_loss,[mu_W_0,rho_W_0,mu_b_0,rho_b_0,mu_W_1,rho_W_1,mu_b_1,rho_b_1])
		
	x  = x.numpy()
	y  = y.numpy()
	z0 = [z0.numpy()]
	a0 = [a0.numpy()]
	y_hat = y_hat.numpy()

	loss = variational_free_energy(W, b, sigmas_W, sigmas_b, eps_W, eps_b, y, y_hat, num_samples)

	print np.linalg.norm(loss-tf_loss)

	gW, gb = W_grad_log_likelihood(W, b, z0, a0, y, y_hat, x, num_samples)
	g_mus_W = mu_grad(gW, W, num_samples)
	g_rhos_W = rho_grad(gW, W, rhos_W, sigmas_W, eps_W, num_samples)
	g_mus_b = mu_grad(gb, b, num_samples)
	g_rhos_b = rho_grad(gb, b, rhos_b, sigmas_b, eps_b, num_samples)
		
	print np.linalg.norm(g_mus_W[0]-g_mu_W_0)
	print np.linalg.norm(g_rhos_W[0]-g_rho_W_0)
	print np.linalg.norm(g_mus_b[0]-g_mu_b_0)
	print np.linalg.norm(g_rhos_b[0]-g_rho_b_0)
	print np.linalg.norm(g_mus_W[1]-g_mu_W_1)
	print np.linalg.norm(g_rhos_W[1]-g_rho_W_1)
	print np.linalg.norm(g_mus_b[1]-g_mu_b_1)
	print np.linalg.norm(g_rhos_b[1]-g_rho_b_1)
	
if __name__ == '__main__':
	grad_check_tf_wb(200, [32])
	grad_check_tf(200, [32])