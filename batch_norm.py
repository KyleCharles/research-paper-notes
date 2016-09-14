def batchnorm_forward(x, gamma, beta, eps):

	N, D = x.shape

	# compute per-dimension mean and std_deviation
	mean = np.mean(x, axis=0)
	var = np.var(x, axis=0)

	# normalize and zero-center (explicit for caching purposes)
	x_mu = x - mean
	inv_var = 1.0 / np.sqrt(var + eps)
	x_hat = x_mu * inv_var

	# squash
	out = gamma*x_hat + beta

	# cache variables for backward pass
	cache = x_mu, inv_var, x_hat, gamma 

	return out, cache

def batchnorm_backward(dout, cache):

	N, D = dout.shape
	x_mu, inv_var, x_hat, gamma = cache

	# intermediate partial derivatives
	dxhat = dout * gamma
	dvar = np.sum((dxhat * x_mu * (-0.5) * (inv_var)**3), axis=0)
	dmu = (np.sum((dxhat * -inv_var), axis=0)) + (dvar * (-2.0 / N) * np.sum(x_mu, axis=0))
	dx1 = dxhat * inv_var
	dx2 = dvar * (2.0 / N) * x_mu
	dx3 = (1.0 / N) * dmu

	# final partial derivatives
	dx = dx1 + dx2 + dx3
	dbeta = np.sum(dout, axis=0)
	dgamma = np.sum(x_hat*dout, axis=0)

	return dx, dgamma, dbeta