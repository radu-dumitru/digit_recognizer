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

### The Euclidean Norm: $`\|v\| = \sqrt{\sum_{i=1}^{10}v_i^2}`$

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

We take the derivative with respect to $`a_j`$: $`\frac{\partial C}{\partial a_j}=\frac{1}{2} \cdot 2 (y_j - a_j) \cdot \frac{\partial(y_j - a_j)}{\partial a_j}`$

Since $`\frac{\partial (y_j - a_j}{\partial a_j}=-1`$

We get $`\frac{\partial C}{\partial a_j}=a_j - y_j`$

```python
def cost_derivative(self, output_activations, y):
	"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
```

# Compute the derivative of the weighted sum from the last layer

We want: $`\frac{\partial C}{\partial z}`$

We know that $`z = w \cdot a_{prev} + b`$

Activations are computed using the sigmoid function: $`a = \sigma(z)`$

We can apply the chain rule: $`\frac{\partial C}{\partial z}=\frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z}`$

$`a=\sigma(z)`$, we differentiate the sigmoid function and we get $`\frac{\partial a}{\partial z}=a(1-a)`$

$`\frac{\partial C}{\partial z}=(a-y) \cdot a(1-a)`$

```python
delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
```

# Compute the derivatives of the weights used by the last layer

Weights affect the cost via the weighted sum: $`z = w \cdot a_{prev} + b`$

We need $`\frac{\partial C}{\partial w}`$

Using the chain rule: $`\frac{\partial C}{\partial w}=\frac{\partial C}{\partial z} \cdot \frac{\partial z}{\partial w}`$

Since $`\frac{\partial C}{\partial z}`$ is known and  $`\frac{\partial z}{\partial w}=a_{prev}`$

We get $`\frac{\partial C}{\partial w}=\frac{\partial C}{\partial z} \cdot a_{prev}`$

```python
nabla_w[-1] = np.dot(delta, activations[-2].transpose())
```

# Compute the derivatives of the biases used by the last layer

We need $`\frac{\partial C}{\partial b}`$

The bias affects the cost function through z: $`\frac{\partial C}{\partial b}=\frac{\partial C}{\partial z} \cdot \frac{\partial z}{\partial b}`$

Since $`\frac{\partial z}{\partial b}=1`$ we get $`\frac{\partial C}{\partial b}=\frac{\partial C}{\partial z}`$ which we already know


```python
nabla_b[-1] = delta
```

# Compute the derivatives of the activation functions from the hidden layer

An activation function from the hidden layer doesn't affect the cost function directly, it first has an effect over all neurons from the output layer which in turn affect the cost function. To find the derivative we need to apply the chain rule:

```math
\frac{\partial C}{\partial a_{hidden}}=\sum_{j}^{} \frac{\partial C}{\partial z_{output,j}} \cdot \frac{\partial z_{output,j}}{\partial a_{hidden}}
```
Since all output neurons depend on $`a_{hidden}`$, we sum over all of them

We have already computed $`\frac{\partial C}{\partial z_{output,j}}`$

We now want to compute $`\frac{\partial z_{output,j}}{\partial a_{hidden}}`$

We know that $`z = w_1a_1 + w_2a_2 + w_3a_3`$

Taking the derivative with respect to one activation function we get: $`\frac{\partial z}{\partial a_1}=w_1`$

$`\frac{\partial C}{\partial a_{hidden}}=\sum_{j}^{} \frac{\partial C}{\partial z_{output,j}} \cdot w_{j}`$

# Compute the derivatives of the z functions from the hidden layer

One z function from the hidden layer doesn't affect the cost function directly but through the activation function that we have computed in the previous step $`\frac{\partial C}{\partial z_{hidden}}=\frac{\partial C}{\partial a_{hidden}} \cdot \frac{\partial a_{hidden}}{\partial z_{hidden}}`$

We know that $`a = \sigma(z)`$

$`\frac{\partial C}{\partial z_{hidden}}=\frac{\partial C}{\partial a_{hidden}} \cdot \sigma'(z)`$

```python
sp = sigmoid_prime(z)
delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
```
Each column represents one of the output neurons and each row contains the weights associated with that neuron
self.weights[-l+1].transpose().shape is (30, 10)

The z functions from the output layer
delta.shape is (10, 1)

The derivatives of the z functions with respect to the activation functions, all from the hidden layer
sp.shape is (30, 1)

Perspective: when multiplying the weights with delta you need to consider the weights as being associated with the neurons from the output layer, look at the output layer and not the hidden layer
