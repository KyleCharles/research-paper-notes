## Large-Batch Training 

**Authors**: *Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang*, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"

**One-liner**: The authors provide numerical evidence that supports the view that large-batch methods tend to converge to sharper minimizers of the training and testing functions leading to poorer generalizations, and discuss several empirical strategies to help alleviate this gap.

### Introduction

In general, our goal when training neural networks is to find the best set of parameters W that will minimize our loss function f. Since this falls under the non-convex optmimization umbrella, we proceed with *iterative refinement*, that is, we refine our set of weights to be slightly better than the previous ones. This is achieved through **Stochastic Gradient Descent** (SGD), or a variant of it called Minibatch Gradient Descent (MGD).

**Why do we use MGD rather than GD or SGD?** 

Well it turns out that computing the gradient over a huge dataset quickly becomes intractable. Updating the parameters using just one training example is another extreme which can oscillate very highly (high variance) and cannot take advantage of highly optimized parallelization routines in the processor, meaning overall training time is increased. 

We thus pick the best of both worlds by sampling a batch (say 256 training examples) that we use to perform a parameter update. This reduces the noise in the parameter update and can lead to more stable convergence. It also allows the computation to take advantage of highly optimized matrix operations if the code is well vectorized.


### Problem Setting

It is to note however that MGD has a major drawback: when batch size is small, there is limited avenue for parallelization, and when batch size is increased, there is a significant loss in generalization performance.


### Meat of the paper

### Experiments

### Conclusion
