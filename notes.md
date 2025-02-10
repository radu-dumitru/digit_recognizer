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

In the equation we see $\|y(x) - a\| ^ 2$, meaning we square the norm, which simplifies to $\|y(x) - a\| ^ 2 = \sum_{i=1}^{10}(y_i - a_i)^2$

This is just the sum of squared differences 

This is a list with all the steps which are expressed by the equation:

- $y$ is a 10-dimensional vector (true label, one-hot encoded)
- $a$ is a 10-dimensional vector (neural network's predicted output)
- Subtract element-wise to get a new 10-dimensional vector
- Apply the Euclidean norm
	- Square each element
	- Sum them
	- Take the square root
- The norm is squared ($\|y(x) - a\| ^ 2 $), so the square root cancels out
- This is done for every training example $x$
- Sum over all training examples
- Divide by $\frac{1}{2n}\$ (where $n$ is the number of training examples)
