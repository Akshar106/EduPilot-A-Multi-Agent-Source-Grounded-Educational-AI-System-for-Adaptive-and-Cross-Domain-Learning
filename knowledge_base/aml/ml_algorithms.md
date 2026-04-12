# Applied Machine Learning — Algorithms and Deep Learning

## Linear Models

### Linear Regression
Linear regression models the relationship between input features x and a continuous target y as a linear function:

**ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ**

The weights w are learned by minimizing the Mean Squared Error (MSE). This is solved analytically using the Normal Equation or iteratively using Gradient Descent.

**Assumptions:** Linear relationship, no multicollinearity, homoscedasticity, normally distributed residuals.

### Logistic Regression
Despite its name, logistic regression is a classification algorithm. It models the probability of class membership using the sigmoid function:

**P(y=1|x) = σ(wᵀx) = 1 / (1 + e^(-wᵀx))**

For values > 0.5, predict class 1; otherwise predict class 0. Trained by minimizing log-loss (binary cross-entropy).

## Tree-Based Models

### Decision Trees
A decision tree recursively partitions the feature space using if/else rules. At each node, it selects the feature and threshold that best separates the classes (measured by Gini impurity or information gain/entropy).

**Pros:** Interpretable, handles mixed data types, no scaling needed.
**Cons:** High variance (prone to overfitting), unstable (small data changes → different trees).

### Random Forests
An ensemble of decision trees trained on random subsets of data (bootstrap sampling) and random subsets of features (random feature selection at each split). Final prediction is majority vote (classification) or average (regression).

**Why it works:**
- Bootstrap sampling → diverse trees
- Random feature selection → less correlated trees
- Averaging → reduces variance while maintaining low bias

**Hyperparameters:** Number of trees, max depth, min samples per leaf, max features.

### Gradient Boosted Trees (GBT)
Boosting builds trees sequentially, where each new tree corrects the errors of the previous ensemble. Gradient boosting minimizes the loss function using gradient descent in function space.

**Popular implementations:** XGBoost, LightGBM, CatBoost.

**Differences from Random Forest:**
- Boosting: sequential, corrects errors, can overfit with too many rounds
- Bagging (RF): parallel, averages independent trees, more robust to overfitting

## Support Vector Machines (SVM)

SVM finds the hyperplane that maximally separates the classes. The **margin** is the distance between the hyperplane and the nearest data points (support vectors). SVM maximizes this margin.

**Soft-margin SVM:** Allows some misclassification (controlled by parameter C):
- High C → smaller margin, fewer errors (more overfitting risk)
- Low C → larger margin, more errors (more regularization)

**Kernel trick:** For non-linearly separable data, SVMs use kernel functions to implicitly map data to higher dimensions:
- Linear kernel
- Polynomial kernel
- Radial Basis Function (RBF/Gaussian) kernel — most common

## Neural Networks

### Feedforward Neural Networks (Multilayer Perceptron)
A neural network consists of layers of interconnected nodes (neurons). Each connection has a weight, and each neuron applies an activation function to its weighted input sum.

**Architecture layers:**
- **Input layer**: Receives raw features
- **Hidden layers**: Learn intermediate representations
- **Output layer**: Produces predictions

**Common activation functions:**
- **ReLU (Rectified Linear Unit)**: f(x) = max(0, x) — most common in hidden layers, prevents vanishing gradients
- **Sigmoid**: f(x) = 1/(1+e^(-x)) — squashes to [0,1], used in binary output
- **Softmax**: Converts outputs to probability distribution (multiclass classification)
- **Tanh**: Squashes to [-1,1], zero-centered

### Backpropagation
Backpropagation is the algorithm for computing gradients in neural networks using the chain rule of calculus. It works backward from the output loss through the network, computing how much each weight contributed to the error.

**Training loop:**
1. Forward pass: compute predictions
2. Compute loss
3. Backward pass: compute gradients via backpropagation
4. Update weights using gradient descent

### Gradient Descent Variants
- **Batch Gradient Descent**: Uses all training examples per update (accurate but slow)
- **Stochastic Gradient Descent (SGD)**: One example per update (fast but noisy)
- **Mini-batch Gradient Descent**: Batch of n examples per update (balance of both) — most used

**Optimizers:**
- **SGD with momentum**: Adds velocity term to smooth oscillations
- **Adam**: Adaptive learning rates per parameter, combines momentum and RMSProp; widely used default

### Deep Learning Architectures
- **Convolutional Neural Networks (CNNs)**: Designed for grid-like data (images). Use convolutional layers to automatically learn spatial features.
- **Recurrent Neural Networks (RNNs)**: Handle sequential data (time series, text). Maintain hidden state across time steps.
- **Long Short-Term Memory (LSTM)**: Improved RNN that handles long-range dependencies using gate mechanisms.
- **Transformers**: Attention-based architecture that replaced RNNs for NLP. Foundation of BERT, GPT, and modern LLMs.

## Model Selection and Hyperparameter Tuning

### Hyperparameter Tuning Methods
- **Grid Search**: Exhaustively try all combinations from a predefined grid. Thorough but expensive.
- **Random Search**: Sample random combinations. More efficient than grid search for many hyperparameters.
- **Bayesian Optimization**: Use a probabilistic model to guide search (e.g., Optuna, Hyperopt).

### The Validation Set vs. Test Set
- **Validation set**: Used to tune hyperparameters and select the best model during development.
- **Test set**: Held out completely until final evaluation. Never used for model selection.

Contaminating the test set (e.g., repeatedly evaluating on it) leads to overly optimistic performance estimates.

## Dimensionality Reduction

### Principal Component Analysis (PCA)
PCA finds the directions (principal components) of maximum variance in the data and projects the data onto these directions.

**Use cases:**
- Remove correlated features
- Visualization of high-dimensional data in 2D/3D
- Noise reduction
- Speed up downstream ML algorithms

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
Nonlinear dimensionality reduction technique primarily used for **visualization** of high-dimensional data. Preserves local neighborhood structure. Not suitable as input to ML models (non-deterministic, no out-of-sample extension).

## Clustering

### K-Means Clustering
Partitions n data points into K clusters by minimizing within-cluster sum of squared distances.

**Algorithm:**
1. Initialize K cluster centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence

**Choosing K:** Elbow method (plot inertia vs K, look for elbow), silhouette score.

**Limitations:** Assumes spherical clusters, sensitive to initialization and outliers.

### DBSCAN (Density-Based Spatial Clustering)
Groups points that are densely packed together; marks low-density points as outliers. Does not require specifying K. Good for clusters of arbitrary shapes.
