## Self-Normalizing Neural Networks

**Authors**: *Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter*

**Gist**: The authors introduce self-normalizing neural networks (SNNs) whose layer activations automatically converge towards zero mean and unit variance and are robust to noise and perturbations. 

**Significance**: Removes the need for the finicky **Batch Normalization** and  permits training deeper and deeper networks in a robust training scheme.

## Picture Says It All

<p align="center">
 <img src="loss.png" width="600px">
</p>

## Activation Function

Uses **Scaled Exponential Linear Units** or SELUs which are defined as follows:

$$
\begin{equation}
  \text{selu}(x) = \lambda 
  \begin{cases}
    x & \text{if $x \gt 0$} \\
    \alpha e^{x} - \alpha  & \text{otherwise}
  \end{cases}
\end{equation}
$$

For the case where we would like zero mean and unit variance, `alpha = 1.6732` and `scale = 1.0507`.

**Properties**

- negative and positive values to control the mean
- derivatives approaching 0 to dampen variance
- slope larger than 1 to increase variance
- continuous curve

```
def selu(x):
	alpha = 1.6732632423543772848170429916717
	lamb = 1.0507009873554804934193349852946
	return lamb * np.where(x > 0., x, alpha * np.exp(x) - alpha)
```

## Weight Initialization

Draw the weights from a Gaussian distribution with mean 0 and variance variance = 1/n.

In python, this is equivalent to setting

```
mu = 0 
sigma = 1.0 / N
W = np.random.normal(mu, sigma, N)
```

## Alpha Dropout

New variant designed for SELU activation. Randomly sets inputs to $\alpha '$ where $\alpha ' = - \lambda \times \alpha$ then performs an affine transformation  with parameters a and b that preserve the self-normalizing property of the activations.

Remember to use the `lambda` and `alpha` values corresponding to zero mean and unit variance. On the other hand, the parameters of the affine transformation can be determined as follows:

```
# dropout params
keep = 0.95
q = 1 - keep

# selu params
alpha = 1.6732632423543772848170429916717
lamb = 1.0507009873554804934193349852946
alpha_p = - alpha * lamb

# affine trans params
prod = q + np.power(alpha_p, 2)*q*(1-q)
a = np.power(prod, -0.5)
b = -prod * (alpha_p*(1-q))
```

And here's a quick example with some NN layers:

```python
def alpha_drop(x, alpha_p=-1.758, keep=0.95):
	q = 1 - keep
	
	# create mask
	ones = np.ones(x.shape)
	idx = np.random.rand(*x.shape) < keep
	mask = ones * idx * alpha_p
	
	# apply mask
	x *= mask
	
	# apply affine transformation (using a and b from before)
	out = x*a +b
	
	return out
	
# example
H = selu(np.dot(W, X) + b)
H_drop = alpha_drop(H)
```