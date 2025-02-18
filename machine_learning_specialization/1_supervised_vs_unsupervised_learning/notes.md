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
\frac{\partial}{\partial w}J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(f(x^{(i)})-y^{(i)})x^{(i)} \\
\frac{\partial}{\partial b}J(w,b)=\frac{1}{m}\sum_{i=1}^{m}(f(x^{(i)})-y^{(i)})
\end{aligned}
```
