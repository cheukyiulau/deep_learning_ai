## Supervised Learning

- Learns from labeled data
- Examples
  * Input is email, output is spam or not (spam filtering)
  * Input is audio, output is text transcript (speech recognition)
  * Input is English, output is spanish (machine translation)
  * Input is image, radar info, output position of other cars (autonomous driving)
- Regression is curve-fitting over a dataset
- Classification
  * Predict only small number of discrete possible outputs
  * There could be many input parameters

## Unsupervised Learning

- Finds structure in unlabeled data
- Clustering groups similar data points together
- Examples
  * Google news grouping articles together (clustering)
  * DNA microarray to categorize people (clustering)
- Anomaly detection finds unusal data points
- Dimensionality reduction compresses data using fewer input parameters

## Jupyter Notebooks

- Used in industry as a sandbox

## Linear Regression with One Variable

- Example house size vs. house price
- Build linear regression through model to predict price as a function of size
- Training set is the data used to train the model
- `x` = input variable feature e.g., house size
- `y` = output variable (target variable) e.g., house price
- `m` = number of training examples
- `(x,y)` = single training example
- $(x^i,y^i)$ = `i`-th training example
- In general we:
  * Obtain training set
  * Run training set on learning algorithm
  * Generate a model i.e., function `f(x)`
  * Use `f(x)` to predict $\hat{y}$
- Linear regression with one variable (univariate linear regression)
```math
f(x)=wx+b
```

## Cost Function

- Squared error cost function
```math
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^i-y^i)^2
J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f(x^i)-y^i)^2
```
- Goal is to miminize the cost function
- We can plot `J(w)` as a function of `w` and visualize its minimum
- We can also plot `J(w,b)` and visualize its minimum
  * This is a 3D plot, and easiest to view as a contour graph

## Gradient Descent

- Algorithm for minimizing any function e.g., any cost function $J(w_1,w_2,...,w_n,b)$
- Start with some initial guess
- Keep changing `w,b` in the direction that most reduces `J(w,b)` until we are at or near a local minimum

## Gradient Descent Algorithm

- Update `w` and `b` *simultaneously*
```math
w=w-\alpha\frac{\partial}{\partial w}J(w,b)
```
  * $\alpha\in(0,1)$ is the learning rate, larger $\alpha$ leads to larger steps
  * $\frac{\partial}{\partial w}J(w,b)$ is the step direction
```math
b=b-\alpha\frac{\partial}{\partial b}J(w,b)
```
- If $\alpha$ is too small, it is computationally inefficient
- If $\alpha$ is too large, you may miss the local minimum
- As we approach local minimum, derivate (and the update step) becomes smaller

## Gradient Descent for Linear Regression

- Solving the derivative terms
```math
\begin{aligned}
\frac{\partial}{\partial w}J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)})-y^{(i)}\right)x^{(i)} \\
\frac{\partial}{\partial b}J(w,b)=\frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)})-y^{(i)}\right)
\end{aligned}
```
- The algorithm is to repeat the following until convergence:
```math
\begin{aligned}
w=w-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)})-y^{(i)}\right)x^{(i)} \\
b=b-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(x^{(i)})-y^{(i)}\right)
\end{aligned}
```
- Remember to update `w` and `b` simultaneously
- Square cost error will always have a single global minimum
- Batch gradient descent uses all training examples

## Multiple Features

- $f(x)=w_{1}x_{1}+w_{2}x_{2}+...w_{n}x_{n}+b$
- $f(\vec{x)}=\vec{w}\cdot\vec{x}+b$
- Above is known as multiple linear regression

## Vectorization

- Take advantage of GPUs
- Without vectorization
```python
f = 0
for j in range(0, n):
  f = f + w[j] * x[j]
f = f + b
```
- With vectorization using NumPy
```python
f = np.dot(w, x) + b
```
- Shorter, and more efficient since NumPy parallelizes operations

## Gradient Descent for Multiple Linear Regression

- Model is $f(\vec{x})=\vec{w}\cdot\vec{x}+b$
- Cost function is $J(\vec{w},b)$
- Update functions become:
```math
w_j=w_j-\alpha\frac{\partial}{\partial w_{j}}J(\vec{w},b) \\
b=b-\alpha\frac{\partial}{\partial b}J(\vec{w},b)
```
- Expanding the partial derivatives:
```math
w_{1}=w_{1}-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(\vec{x}^(i))-y^{(i)}\right)x_{1}^{(i)} \\
... \\
w_{n}=w_{n}-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(\vec{x}^(i))-y^{(i)}\right)x_{n}^{(i)}
b=b-\alpha\frac{1}{m}\sum_{i=1}^{m}\left(f(\vec{x}^{(i)})-y^{(i)}\right)
```
- Simultaneously update $w_j$ and $b$
- Note that we can also use normal equation to solve for $w$ and $b$ without iterations
  * Does not generalize to other learning algorithms
  * Slow when the number of features is large
  * May be used in ML libraries that implement linear regression

## Feature Scaling

- Improves gradient descent performance
- Example
  * $p=w_{1}x_{1}+w_{2}x_{2}$
  * $x_{1}\in(300,2000)$
  * $x_{2}\in(0,5)$
- Larger features $x$ will get smaller parameters $w$ and vice-versa
- Causes contour plot of $J(w,b)$ to be stretched (oval) resulting in more iterations to find minimum
- We can scale `x` so that each go from $(0,1)$ resulting in circular $J(w,b)$ contours
- Scaling above example by the max
  * $x_{1}=\frac{x_{1}}{2000}$
  * $x_{2}=\frac{x_{2}}{5}$
- Scaling by mean normalization around zero where $\mu$ is the mean
  * $x_{1}=\frac{x_{1}-\mu_{1}}{2000-300}$
  * $x_{2}=\frac{x_{2}-\mu_{2}}{5-0}$
- Scaling by z-score normalization where $\sigma$ is standard deviation
  * $x_{1}=\frac{x_{1}-\mu_{1}}{\sigma_{1}}$
  * $x_{2}=\frac{x_{2}-\mu_{2}}{\sigma_{2}}$
- In general aim for $-1\le x\le 1$ for each feature
