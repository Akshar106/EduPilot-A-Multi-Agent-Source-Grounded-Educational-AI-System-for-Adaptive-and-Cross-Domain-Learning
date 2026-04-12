# Applied Machine Learning — Fundamentals

## What is Machine Learning?

Machine learning (ML) is a branch of artificial intelligence that enables computer systems to learn from data and improve their performance on tasks without being explicitly programmed. Instead of writing hand-coded rules, a machine learning model learns patterns from training examples and uses those patterns to make predictions or decisions on new, unseen data.

The core idea is to find a function f(x) ≈ y, where x is the input (features) and y is the output (label or target). The model learns the parameters of this function by minimizing a loss function that measures how wrong the predictions are.

## Types of Machine Learning

### Supervised Learning
The model learns from labeled training data — examples where the correct output is known. The goal is to generalize this mapping to new inputs.

**Regression**: Predicting a continuous value (e.g., predicting house prices from features like square footage, location, number of bedrooms).

**Classification**: Predicting a discrete class label (e.g., spam vs. not-spam, cat vs. dog).

Examples of algorithms: Linear regression, logistic regression, decision trees, random forests, support vector machines (SVMs), neural networks.

### Unsupervised Learning
The model learns patterns from unlabeled data. There are no predefined correct answers.

**Clustering**: Grouping similar data points together. Example: K-means clustering for customer segmentation.

**Dimensionality reduction**: Compressing high-dimensional data while preserving important structure. Example: Principal Component Analysis (PCA).

**Density estimation**: Modeling the underlying probability distribution of data.

### Reinforcement Learning
An agent learns to make decisions by interacting with an environment and receiving rewards or penalties. Examples: game-playing AI (chess, Go), robotics control.

## The Bias-Variance Tradeoff

The **bias-variance tradeoff** is one of the most fundamental concepts in machine learning. It describes the tradeoff between two sources of error in a predictive model.

### Bias
**Bias** is the error introduced by approximating a complex real-world problem with a simplified model. A high-bias model makes strong assumptions about the data (e.g., assuming the data is linearly separable when it is not).

- High bias → model is too simple → **underfitting**
- The model systematically misses the true relationship in the data
- High training error and high test error

Example: Using a simple linear model to fit data that has a clear quadratic relationship. The straight line cannot capture the curve, leading to systematic errors on both training and test data.

### Variance
**Variance** is the sensitivity of the model to small fluctuations in the training data. A high-variance model is very flexible and memorizes the training data, including its noise.

- High variance → model is too complex → **overfitting**
- The model performs well on training data but poorly on test data
- Low training error but high test error

Example: A very deep decision tree that grows until every training point is correctly classified. It has memorized the training set, including its random noise, so it fails on new data.

### The Tradeoff
The total expected error of a model can be decomposed as:

**Total Error = Bias² + Variance + Irreducible Noise**

- **Irreducible noise** is inherent randomness in the data that no model can eliminate.
- As model complexity increases: Bias decreases, Variance increases.
- As model complexity decreases: Bias increases, Variance decreases.

The sweet spot is a model that is complex enough to capture the true signal (low bias) but not so complex that it fits the noise (low variance).

**How to detect it:**
- Underfitting (high bias): Training error and test error are both high.
- Overfitting (high variance): Training error is low, but test error is high (large gap).
- Good fit: Both training and test errors are low and close to each other.

## Overfitting and Underfitting

### Overfitting
Overfitting occurs when a model learns the training data too well, including its noise and random fluctuations, resulting in poor generalization to new data.

**Signs of overfitting:**
- Very low training loss, high validation/test loss
- Large gap between training and validation performance
- Model performs significantly worse in production than in development

**Causes:**
- Model is too complex relative to the amount of training data
- Training for too many epochs
- Too many features with insufficient data
- Insufficient regularization

**Solutions:**
1. **Regularization** (L1/L2): Adds a penalty term to the loss function to discourage large weights
2. **Dropout**: Randomly deactivates neurons during training in neural networks
3. **Early stopping**: Stop training when validation loss starts to increase
4. **Data augmentation**: Artificially increase the training dataset
5. **Cross-validation**: Use k-fold CV to get a more robust performance estimate
6. **Reduce model complexity**: Prune decision trees, reduce network depth

### Underfitting
Underfitting occurs when the model is too simple to capture the underlying patterns in the data.

**Signs of underfitting:**
- High training loss (poor performance even on training data)
- Model fails to capture obvious trends

**Solutions:**
1. Use a more complex model (more layers, more features, higher-degree polynomial)
2. Add more relevant features (feature engineering)
3. Reduce regularization strength
4. Train for more epochs

## Regularization

Regularization techniques constrain model complexity to prevent overfitting.

### L2 Regularization (Ridge)
Adds the sum of squared weights to the loss function:

**Loss = Original Loss + λ × Σ(wᵢ²)**

- λ (lambda) controls regularization strength
- Penalizes large weights, pushing them toward zero (but not exactly zero)
- Used in Ridge Regression, neural networks with weight decay

### L1 Regularization (Lasso)
Adds the sum of absolute values of weights:

**Loss = Original Loss + λ × Σ|wᵢ|**

- Encourages sparsity — many weights become exactly zero
- Useful for feature selection (irrelevant features get weights zeroed out)
- Used in Lasso Regression

### Elastic Net
Combines both L1 and L2:

**Loss = Original Loss + λ₁ × Σ|wᵢ| + λ₂ × Σ(wᵢ²)**

### Dropout (for Neural Networks)
During training, randomly set a fraction of neurons to zero with probability p (the dropout rate). This forces the network to learn redundant representations and reduces co-adaptation of neurons.

## Model Evaluation Metrics

### For Classification
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) — fraction of correct predictions
- **Precision**: TP / (TP + FP) — of all predicted positives, how many are truly positive
- **Recall (Sensitivity)**: TP / (TP + FN) — of all actual positives, how many were found
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve; measures discriminative power

### For Regression
- **Mean Squared Error (MSE)**: Average of squared differences between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **Mean Absolute Error (MAE)**: Average of absolute differences
- **R² (R-squared)**: Proportion of variance explained by the model (1 = perfect, 0 = no better than predicting the mean)

## Cross-Validation

Cross-validation is a technique for evaluating model performance more reliably by using multiple train/test splits.

### K-Fold Cross-Validation
1. Split the dataset into K equal-sized folds
2. For each fold: use it as the test set and train on the remaining K-1 folds
3. Average the performance metrics across all K runs

Typical K values: 5 or 10. Larger K = more reliable estimate but more computation.

**Benefits:**
- Uses all data for both training and evaluation
- Provides variance estimate of performance
- Reduces overfitting to a specific train/test split

### Stratified K-Fold
For classification: ensures each fold has the same class proportions as the full dataset. Important for imbalanced datasets.

## Feature Engineering

Feature engineering is the process of transforming raw data into features that better represent the underlying patterns, improving model performance.

**Common techniques:**
- **Normalization/Standardization**: Scale features to [0,1] or mean=0, std=1
- **One-hot encoding**: Convert categorical variables to binary columns
- **Log transformation**: Reduce skewness in right-skewed distributions
- **Polynomial features**: Add x², x³ terms to capture non-linear relationships
- **Interaction features**: Multiply two features to capture their joint effect
- **Missing value imputation**: Fill NaN values with mean, median, or predicted values
