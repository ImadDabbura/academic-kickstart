---
title: Coding Neural Network - Gradient Checking
date: 2018-04-08
math: true
diagram: true
markup: mmark
image:
  placement: 3
tags: ["Deep Learning", "AI"]
categories: ["Machine Learning", "Deep Learning"]
---

In the previous post, [*Coding Neural Network - Forward Propagation and Backpropagation*](https://imaddabbura.github.io/blog/machine%20learning/deep%20learning/2018/04/01/coding-neural-network-fwd-back-prop.html), we implemented both forward propagation and backpropagation in `numpy`. However, implementing backpropagation from scratch is usually more prune to bugs/errors. Therefore, it's necessary before running the neural network on training data to check if our implementation of backpropagation is correct. Before we start, let's revisit what back-propagation is: We loop over the nodes in reverse topological order starting at the final node to compute the derivative of the cost with respect to each edge's node tail. In other words, we compute the derivative of cost function with respect to all parameters, i.e $\frac{\partial J}{\partial \theta}$ where $\theta$ represents the parameters of the model.

The way to test our implementation is by computing numerical gradients and compare it with gradients from backpropagation (analytical). There are two way of computing numerical gradients:

- Right-hand form:

$$\frac{J(\theta + \epsilon) - J(\theta)}{\epsilon}\tag{1}$$

- Two-sided form (see figure 2):

$$\frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2 \epsilon}\tag{2}$$

{{< figure library="1" src="coding-nn-from-scratch/two_sided_gradients.png" title="Two-sided numerical gradients." numbered="true">}}

Two-sided form of approximating the derivative is closer than the right-hand form. Let's illustrate that with the following example using the function $f(x) = x^2$ by taking its derivative at $x = 3$.

- Analytical derivative:

$$\nabla_x f(x) = 2x\ \Rightarrow\nabla_x f(3) = 6$$

- Two-sided numerical derivative:

$$\frac{(3 + 1e-2)^2 - (3 - 1e-2)^2}{2 * 1e-2} = 5.999999999999872$$

- Right-hand numerical derivative:

$$\frac{(3 + 1e-2)^2 - 3^2}{1e-2} = 6.009999999999849$$

As we see above, the difference between analytical derivative and two-sided numerical gradient is almost zero; however, the difference between analytical derivative and right-sided derivative is 0.01. Therefore, we'll use two-sided epsilon method to compute the numerical gradients.

In addition, we'll normalize the difference between numerical. gradients and analytical gradients using the following formula:

$$\frac{\|grad - grad\_{approx}\|_2}{\|grad\|_2 + \|grad\_{approx}\|_2}\tag{3}$$

If the difference is $\leq 10^{-7}$, then our implementation is fine; otherwise, we have a mistake somewhere and have to go back and revisit backpropagation code.

Below are the steps needed to implement gradient checking:

1. Pick random number of examples from training data to use it when computing both numerical and analytical gradients.
    - Don't use all examples in the training data because gradient checking is very slow.
2. Initialize parameters.
3. Compute forward propagation and the cross-entropy cost.
4. Compute the gradients using our back-propagation implementation.
5. Compute the numerical gradients using the two-sided epsilon method.
6. Compute the difference between numerical and analytical gradients.

We'll be using functions we wrote in *"Coding Neural Network - Forward Propagation and Backpropagation"* post to initialize parameters, compute forward propagation and back-propagation as well as the cross-entropy cost.

Let's first import the data.

```python
# Loading packages
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import seaborn as sns

sys.path.append("../scripts/")
from coding_neural_network_from_scratch import (initialize_parameters,
                                                L_model_forward,
                                                L_model_backward,
                                                compute_cost)
```

```python
# Import the data
train_dataset = h5py.File("../data/train_catvnoncat.h5")
X_train = np.array(train_dataset["train_set_x"]).T
y_train = np.array(train_dataset["train_set_y"]).T
X_train = X_train.reshape(-1, 209)
y_train = y_train.reshape(-1, 209)

X_train.shape, y_train.shape
```
    ((12288, 209), (1, 209))

Next, we'll write helper functions that faciltate converting parameters and gradients dictionaries into vectors and then re-convert them back to dictionaries.

```python
def dictionary_to_vector(params_dict):
    count = 0
    for key in params_dict.keys():
        new_vector = np.reshape(params_dict[key], (-1, 1))
        if count == 0:
            theta_vector = new_vector
        else:
            theta_vector = np.concatenate((theta_vector, new_vector))
        count += 1

    return theta_vector


def vector_to_dictionary(vector, layers_dims):
    L = len(layers_dims)
    parameters = {}
    k = 0

    for l in range(1, L):
        # Create temp variable to store dimension used on each layer
        w_dim = layers_dims[l] * layers_dims[l - 1]
        b_dim = layers_dims[l]

        # Create temp var to be used in slicing parameters vector
        temp_dim = k + w_dim

        # add parameters to the dictionary
        parameters["W" + str(l)] = vector[
            k:temp_dim].reshape(layers_dims[l], layers_dims[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    # Get the number of indices for the gradients to iterate over
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads)// 2
    count = 0
    
    # Iterate over all gradients and append them to new_grads list
    for l in range(1, L + 1):
        if count == 0:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)))
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        count += 1
        
    return new_grads
```

Finally, we'll write the gradient checking function that will compute the difference between the analytical and numerical gradients and tell us if our implementation of back-propagation is correct. We'll randomly choose 1 example to compute the difference.

```python
def forward_prop_cost(X, parameters, Y, hidden_layers_activation_fn="tanh"):
    # Compute forward prop
    AL, _ = L_model_forward(X, parameters, hidden_layers_activation_fn)

    # Compute cost
    cost = compute_cost(AL, Y)

    return cost


def gradient_check(
        parameters, gradients, X, Y, layers_dims, epsilon=1e-7,
        hidden_layers_activation_fn="tanh"):
    # Roll out parameters and gradients dictionaries
    parameters_vector = dictionary_to_vector(parameters)
    gradients_vector = gradients_to_vector(gradients)

    # Create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(parameters_vector)

    for i in range(len(parameters_vector)):
        # Compute cost of theta + epsilon
        theta_plus = np.copy(parameters_vector)
        theta_plus[i] = theta_plus[i] + epsilon
        j_plus = forward_prop_cost(
            X, vector_to_dictionary(theta_plus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute cost of theta - epsilon
        theta_minus = np.copy(parameters_vector)
        theta_minus[i] = theta_minus[i] - epsilon
        j_minus = forward_prop_cost(
            X, vector_to_dictionary(theta_minus, layers_dims), Y,
            hidden_layers_activation_fn)

        # Compute numerical gradients
        grads_approx[i] = (j_plus - j_minus) / (2 * epsilon)

    # Compute the difference of numerical and analytical gradients
    numerator = norm(gradients_vector - grads_approx)
    denominator = norm(grads_approx) + norm(gradients_vector)
    difference = numerator / denominator

    if difference > 10e-7:
        print ("\033[31mThere is a mistake in back-propagation " +\
               "implementation. The difference is: {}".format(difference))
    else:
        print ("\033[32mThere implementation of back-propagation is fine! "+\
               "The difference is: {}".format(difference))

    return difference
```

```python
# Set up neural network architecture
layers_dims = [X_train.shape[0], 5, 5, 1]

# Initialize parameters
parameters = initialize_parameters(layers_dims)

# Randomly selecting 1 example from training data
perms = np.random.permutation(X_train.shape[1])
index = perms[:1]

# Compute forward propagation
AL, caches = L_model_forward(X_train[:, index], parameters, "tanh")

# Compute analytical gradients
gradients = L_model_backward(AL, y_train[:, index], caches, "tanh")

# Compute difference of numerical and analytical gradients
difference = gradient_check(parameters, gradients, X_train[:, index], y_train[:, index], layers_dims)
```

There implementation of back-propagation is fine! The difference is: 3.0220555297630148e-09

Congratulations! Our implementation is correct :thumbsup:

<h2 style="font-family: Georgia; font-size:2em;color:purple; font-style:bold">
Conclusion
</h2>

Below are some key takeaways:

- Two-sided numerical gradient approximates the analytical gradients more closely than right-side form.
- Since gradient checking is very slow:
  - Apply it on one or few training examples.
  - Turn it off when training neural network after making sure that backpropagation's implementation is correct.
- Gradient checking doesn't work when applying drop-out method. Use keep-prob = 1 to check gradient checking and then change it when training neural network.
- Epsilon = $10e-7$ is a common value used for the difference between analytical gradient and numerical gradient. If the difference is less than 10e-7 then the implementation of backpropagation is correct.
- Thanks to *Deep Learning* frameworks such as Tensorflow and Pytorch, we may find ourselves rarely implement backpropagation because such frameworks compute that for us; however, it's a good practice to understand what happens under the hood to become a good Deep Learning practitioner.

The source code that created this post can be found [here](https://github.com/ImadDabbura/blog-posts/blob/master/notebooks/Coding-Neural-Network-Gradient-Checking.ipynb).
