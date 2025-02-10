# The Cost Function

```math
C(w, b) = \frac{1}{2n}\sum_{x}^{} \|y(x) - a\| ^ 2
```
$w$ - weights

$b$ - biases

$n$ - the number of training examples

$x$ - a single input image

$y(x)$ - the desired output. It is a 10 dimensional vector (one-hot encoded)

$a$ - the output of the neural network. It is a 10 dimensional vector which contain the probability distribution over the 10 possible digits.

We subtract the 2 vectors element-wise: $`v = y - a`$. The result is a 10 dimensional vector.

### The Euclidean Norm
```math
\|v\| = \sqrt{\sum\limits_{i=1}^{10} v_i^2}
```
