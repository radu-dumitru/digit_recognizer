# We are going to assume an architecture with an input layer made up of 768 neurons, a hidden layer made up of 30 neurons and an output layer which contains 10 neurons (768, 30, 10)

# The Cost Function

```math
C(w, b) = \frac{1}{2n}\sum_{x}^{} \|y(x) - a\| ^ 2
```
$w$ - weights

$b$ - biases

$n$ - the number of training examples

$\sum_{x}^{}$ - sum over all training inputs x in the dataset

$x$ - a single input image

$y(x)$ - the desired output. It is a 10 dimensional vector (one-hot encoded)

$a$ - the output of the neural network. It is a 10 dimensional vector which contain the probability distribution over the 10 possible digits.

We subtract the 2 vectors element-wise: $`v = y - a`$. The result is a 10 dimensional vector.

### The Euclidean Norm
```math
\|v\| = \sqrt{\sum_{i=1}^{10}v_i^2}
```

In the equation we see $`\|y(x) - a\|^2`$, meaning we square the norm, which simplifies to $`\|y(x) - a\|^2 = \sum_{i=1}^{10}(y_i - a_i)^2`$

This is just the sum of squared differences 

This is a list with all the steps which are expressed by the equation:

- $y$ is a 10-dimensional vector (true label, one-hot encoded)
- $a$ is a 10-dimensional vector (neural network's predicted output)
- Subtract element-wise to get a new 10-dimensional vector
- Apply the Euclidean norm
	- Square each element
	- Sum them
	- Take the square root
- The norm is squared ($`\|y(x) - a\|^2`$), so the square root cancels out
- This is done for every training example $x$
- Sum over all training examples
- Divide by $\frac{1}{2n}\$ (where $n$ is the number of training examples)

When applying backpropagation, we process one training example at a time to compute the gradients of the cost function with respect to the weights and biases. This means the cost function for a single example has the following form:

```math
C = \frac{1}{2}\sum_{j}^{} (y_j - a_j) ^ 2
```

# Compute the Derivative of Cost w.r.t. Output Activations 

We need this derivative: $`\frac{\partial C}{\partial a_j}`$

We start from: $`C = \frac{1}{2}\sum_{j}^{} (y_j - a_j) ^ 2`$

We are taking the derivative with respect to $`a_j`$. This means that all the other temrs in the summation are constant and their derivatives are 0. So we can focus on the term that contains $`a_j`$: $`C = \frac{1}{2}(y_j - a_j)^2`$

We take the derivative with respect to $`a_j`$: $`\frac{\partial C}{\partial a_j}=\frac{1}{2} \ast2 (y_j - a_j) \ast \frac{\partial(y_j - a_j)}{\partial a_j}`$

Since $`\frac{\partial (y_j - a_j}{\partial a_j}=-1`$

We get $`\frac{\partial C}{\partial a_j}=a_j - y_j`$


