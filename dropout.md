## Dropout

**Authors**: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014

**One-liner**: The authors address the problem of *overfitting* by introducing *dropout* during neural network training.

### Problem Setting

Deep neural networks have multiple non-linear hidden layers which means there is a tendency for them to overfit when the training data is limited. Furthermore, techniques like ensembling - meant to regularize these networks - are usually computationally intractable due to training and  hyperparameter tuning.

The best way to regularize a neural network would be to average the predictions of all possible settings of the parameters and weigh each setting by its posterior probability, a computational power we don't have.

### Meat of the paper

Dropout addresses both these issues:

- it prevents overfitting
- it approximately combines exponentially different neural network architectures as a form of ensembling.

Basically, dropout means keeping a neuron active with some probability **p**, or setting it to zero otherwise during training. Note that **p** is a hyperparameter which can be cross-validated but it's usually set to 0.5. 

Below is an image taken from the paper illustrating dropout.

<p align="center">
 <img src="/img/dropout/dropout_schema.png" alt="Drawing" width="550px">
</p>

**So how is this a form of ensembling?**

The authors describe dropout as sampling a *thinned* network from the original one, and since a neural net is a collection of 2^n possible thinned networks, then every time we train with dropout, we can view it as training a collection of 2^n thinned networks with weight sharing (most of the thinned networks don't get trained though).

### Train vs. Test

Note that at test time, we don't average the prediction from all the thinned models - this isn't practical!

Instead, we quit using dropout and scale our hidden layers by **p**. In this manner, the output of our neurons at test time have the same expected output that they had at training time. This scaling is very important because it enables us to combine the 2^n networks with shared weights into a single neural network.

<p align="center">
 <img src="/img/dropout/scaling.png" alt="Drawing" width="550px">
</p>

### Motivation

I think that this is the most beautiful part of the paper. The authors use the idea of the superiority of sexual reproduction over asexual reproduction as motivation for dropout.

In fact, sexual reproduction reduces complex co-adaptations between genes by forcing certain ones to be useful on their own or in collaboration with a small number of other genes rather than relying on the work of collective whole. This imbues sexual reproduction offsprings with a certain robustness: they have the ability to adapt to changes in the environment. In the same manner, dropout makes the hidden units in a neural network more robust and drives them towards creating useful features on their own.

By the way, the authors also describe a similar motivation using the idea of *conspiracies*.

### Conclusion

Advantages:

- acts as a form of regularization
- acts as a form of ensembling
- extendable to other graphical models such as RBM's

Drawbacks:

- increase in training time
- increase in test time - use inverted dropout instead as mentioned in [CS231n](http://cs231n.github.io/neural-networks-2/#reg)
