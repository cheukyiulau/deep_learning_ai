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
  * Use `f(x)` to predict $\hat(y)$