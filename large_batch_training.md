## Large-Batch Training 

**Authors**: *Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang*, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"

**One-liner**: The authors provide numerical evidence that supports the view that large-batch methods tend to converge to sharper minimizers of the training and testing functions leading to poorer generalizations, and then discuss several empirical strategies to help alleviate this gap.

### Introduction

In general, the goal when training neural networks is to find the best set of parameters W that will minimize our loss function f. Since this falls under the non-convex optmimization umbrella, we proceed with *iterative refinement*, that is, we continuously update our set of weights to be slightly better than the previous ones. This is achieved through **Stochastic Gradient Descent** (SGD), or a variant of it called Minibatch Gradient Descent (MGD).

*Why do we use MGD rather than GD or SGD?*

Well it turns out that computing the gradient over a huge dataset is intractable. Updating the parameters using just one training example is another extreme which can oscillate very highly (high variance) and cannot take advantage of highly optimized parallelization routines in the processor, meaning overall training time is increased. 

We thus pick the **best of both worlds** by sampling a batch (say 256 training examples) that we use to perform a parameter update. This reduces the noise in the parameter update and can lead to more stable convergence. It also allows the computation to take advantage of highly optimized matrix operations if the code is well vectorized.


### Problem Setting

MGD suffers from an annoying dichotomy: when batch size is small, there is limited avenue for parallelization (slow training and convergence), and when batch size is increased, there is a significant loss in generalization performance (higher test error). 

So why does an increase in batch size correlate to a higher generalization gap?

This is the question the authors shed light upon. 

### Meat of the Paper

A. Drawbacks of Large-Batch Methods

The experiments and data presented by the authors support 2 conjectures that try to explain the gap in generalization between LB and SB. They are:

- LB methods lack the explorative properties of SB methods and tend to zoom in on the minimizer closest to the initial point.
- SB and LB methods converge to qualitatively different minimizers with differing generalization properties.

Basically what this means is that LB methods converge to sharp minimizers of the training function, which are characterized by large positive eigenvalues in the Hessian as illustrated below:

<p align="center">
 <img src="/img/large_batch/minimizers.png" width="480px">
</p>

While the training loss is seemingly low, test error is much higher, hence a gap in the generalization ability of the model.

B. Success of Small-Batch Methods

This begs the question of why do SB methods perform better, or put differently, **why do they not converge to sharp minimizers**?

According to the authors, SB method gradients are noisier than their LB counterparts. This pushes the iterate out of the minimizer basin and tugs it towards flatter minimizers where noise will not cause an exit. If the initial noise had been smaller, the iterate would have stayed at the sharper minimizer.


### Experiments

Settings:

- Parametric 1D plots ([Goodfellow et al.](https://arxiv.org/abs/1412.6544)) for vizualization
- Metric of sharpness 

<p align="center">
 <img src="/img/large_batch/metric.png" width="480px">
</p>
- models: 6 different network architectures

<p align="center">
 <img src="/img/large_batch/archs.png" width="480px">
</p>

- loss: mean cross entropy
- LB uses 10% of training data for the batch size, SB uses 256 samples
- Adam update rule


Curved path between two points result:

<p align="center">
 <img src="/img/large_batch/curvi.png" width="480px">
</p>

Linear slice between two points result:

<p align="center">
 <img src="/img/large_batch/linear.png" width="480px">
</p>

**Legend**: 

- Left axis is cross-entropy loss, right axis is classification accuracy. 
- Solid line is training data set, dashed line is testing data set. 
- `alpha = 0` corresponds to SB minimizer, `alpha=1` corresponds to LB minimizer.

**Observation**: LB minima are strikingly sharper than the SB minima in the one-dimensional manifold.


### Proposed Solutions

To alleviate large-batch gradient descent problems, the authors propose 3 potential avenues of improvement:

- **data augmentation**: accuracy improves but sharpness still exists. Could be problematic...
- **robust training**: attempts to lower an `epsilon` disc down the loss surface rathee than finding the low point in the valley ==> does not improve generalization outcomes, maybe try adversarial training?
- **conservative training**: agin, there is a statistically significant improvement in the testing accuracy of the large-batch method but does not solve the problem of sensitivity (to images contained in neither training or testing set).

### Conclusion

By providing evidence that the "sharpness" measured in 1-D parametric plots of multiple deep learning architectures increases with larger batches, the authors were able to show that sharp minimizers do in fact cause a deterioration in generalization.

An interesting method that could potentially alleviate this problem would be to use a switching-based approach while training: leverage small batches in the first few epochs and then transition to larger batch methods.
