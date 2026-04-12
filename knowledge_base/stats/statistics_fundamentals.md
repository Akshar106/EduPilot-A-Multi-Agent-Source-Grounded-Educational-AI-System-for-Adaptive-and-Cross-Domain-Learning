# Statistics — Fundamentals

## Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset.

### Measures of Central Tendency
- **Mean (Average)**: Sum of all values divided by the count. Sensitive to outliers.
  - Population mean: μ = Σxᵢ / N
  - Sample mean: x̄ = Σxᵢ / n

- **Median**: The middle value when data is sorted. Robust to outliers. Preferred for skewed data.

- **Mode**: The most frequently occurring value. Can be used for categorical data.

### Measures of Spread
- **Variance**: Average of squared deviations from the mean.
  - Population variance: σ² = Σ(xᵢ - μ)² / N
  - Sample variance: s² = Σ(xᵢ - x̄)² / (n-1)  ← uses n-1 (Bessel's correction)

- **Standard Deviation**: Square root of variance. In the same units as the data.
  - σ (population), s (sample)

- **Range**: Max − Min. Simple but sensitive to outliers.

- **Interquartile Range (IQR)**: Q3 − Q1. The range of the middle 50% of data. Robust to outliers.

- **Coefficient of Variation (CV)**: (s / x̄) × 100%. Relative spread; allows comparison across different units.

### Shape of Distributions
- **Skewness**: Measures asymmetry.
  - Positive (right) skew: tail extends to the right; mean > median
  - Negative (left) skew: tail extends to the left; mean < median
  - Symmetric: mean ≈ median ≈ mode

- **Kurtosis**: Measures the heaviness of the tails.
  - Leptokurtic (high kurtosis): heavy tails, sharper peak
  - Platykurtic (low kurtosis): light tails, flatter peak
  - Mesokurtic: normal distribution (kurtosis = 3)

## Probability Theory

### Basic Probability Rules
- **Probability of an event A**: P(A), where 0 ≤ P(A) ≤ 1
- **Complement**: P(Aᶜ) = 1 − P(A)
- **Addition rule**: P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
- **Multiplication rule (independent events)**: P(A ∩ B) = P(A) × P(B)

### Conditional Probability
P(A|B) = P(A ∩ B) / P(B)

The probability of A given that B has occurred.

### Bayes' Theorem
P(A|B) = [P(B|A) × P(A)] / P(B)

**Components:**
- P(A): Prior probability of A
- P(B|A): Likelihood — probability of observing B if A is true
- P(B): Marginal probability of B
- P(A|B): Posterior probability of A given B

**Example:** Medical testing. If a disease has prevalence 1%, a test has 99% sensitivity (true positive rate) and 95% specificity (true negative rate):
P(Disease|Positive Test) = (0.99 × 0.01) / [(0.99 × 0.01) + (0.05 × 0.99)] ≈ 0.167

Only 16.7% of positive tests actually have the disease! This is due to the low base rate (prior probability).

## Probability Distributions

### Normal Distribution (Gaussian)
The most important distribution in statistics due to the Central Limit Theorem.

**Parameters:** Mean μ (location) and standard deviation σ (spread)
**PDF:** f(x) = (1/σ√2π) × e^[-(x-μ)²/2σ²]

**Properties:**
- Bell-shaped, symmetric about the mean
- 68% of data within 1σ, 95% within 2σ, 99.7% within 3σ (68-95-99.7 rule)
- Mean = Median = Mode

**Standard Normal (Z):** μ = 0, σ = 1. Any normal can be standardized: Z = (X - μ) / σ

### Binomial Distribution
Models the number of successes in n independent trials, each with probability p of success.

**Parameters:** n (trials), p (success probability)
**PMF:** P(X = k) = C(n,k) × pᵏ × (1-p)^(n-k)
**Mean:** μ = np
**Variance:** σ² = np(1-p)

**Example:** In 20 coin flips with a fair coin, P(X = 10) using the binomial formula.

### t-Distribution (Student's t)
Used when estimating the population mean from a small sample or when the population standard deviation is unknown.

**Properties:**
- Symmetric and bell-shaped like the normal distribution
- Heavier tails than normal (more probability in extreme values)
- Approaches the normal distribution as degrees of freedom (df = n - 1) increase
- For n ≥ 30, t-distribution is very similar to the standard normal

### Chi-Square Distribution (χ²)
Distribution of the sum of squares of k independent standard normal variables. Used in:
- Goodness-of-fit tests
- Test of independence (contingency tables)
- Confidence intervals for variance

### F-Distribution
Ratio of two chi-squared variables. Used in:
- ANOVA (comparing means of 3+ groups)
- Testing equality of variances

## Central Limit Theorem (CLT)

**Statement:** If you draw samples of size n from any population with mean μ and finite variance σ², the distribution of the sample mean x̄ approaches a normal distribution as n → ∞.

Specifically: x̄ ~ N(μ, σ²/n) for large n (rule of thumb: n ≥ 30)

**Implications:**
- Justifies using normal-distribution-based methods even for non-normal populations
- The standard error of the mean (SEM) = σ/√n decreases as sample size increases
- Foundation of most inferential statistics

**Why it matters for ML:** When we compute training loss averaged over many samples, the CLT tells us it will be approximately normally distributed, justifying many statistical tests.

## Confidence Intervals

A **confidence interval (CI)** is a range of values that is likely to contain the true population parameter with a given level of confidence.

### Definition
A 95% confidence interval means: if we repeated the sampling procedure many times and computed a CI each time, 95% of those intervals would contain the true population parameter.

**Important note:** The 95% refers to the procedure, not any single interval. Once computed, a specific interval either contains the true parameter or it doesn't.

### CI for the Population Mean (known σ)
x̄ ± z(α/2) × (σ/√n)

For 95% CI: z(0.025) = 1.96
For 99% CI: z(0.005) = 2.576

### CI for the Population Mean (unknown σ, small sample)
x̄ ± t(α/2, n-1) × (s/√n)

Use the t-distribution with n-1 degrees of freedom. For large n, this approaches the z-interval.

### Interpretation
"I am 95% confident that the true population mean lies between [lower bound] and [upper bound]."

### Width of CI
- Wider CI → less precise but more likely to capture the true parameter
- Narrower CI → more precise but requires larger sample size or higher variance tolerance
- Width decreases with: larger sample size n, smaller σ, lower confidence level

### Margin of Error
E = z(α/2) × (σ/√n)

The CI is x̄ ± E. Doubling precision (halving margin of error) requires quadrupling sample size.

## Correlation and Covariance

### Covariance
Measures how two variables vary together:
Cov(X, Y) = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / (n-1)

Positive covariance: variables tend to increase together.
Negative covariance: one variable increases as the other decreases.

### Pearson Correlation Coefficient
r = Cov(X, Y) / (sₓ × sᵧ), where r ∈ [-1, 1]

- r = 1: Perfect positive linear relationship
- r = -1: Perfect negative linear relationship
- r = 0: No linear relationship (may still have non-linear relationship)

**Correlation does not imply causation.** Ice cream sales and drowning rates are positively correlated because both increase in summer, not because ice cream causes drowning.

### Spearman Rank Correlation
Non-parametric alternative. Based on ranks rather than values. Captures monotonic (not just linear) relationships. Robust to outliers and non-normal distributions.

## Simple Linear Regression

Models the linear relationship between a predictor X and an outcome Y:

**Y = β₀ + β₁X + ε**

Where:
- β₀ = intercept (predicted Y when X = 0)
- β₁ = slope (change in Y per unit increase in X)
- ε = random error term

### Ordinary Least Squares (OLS)
Estimates β₀ and β₁ by minimizing the sum of squared residuals:

β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ(xᵢ - x̄)² = Cov(X,Y) / Var(X)
β₀ = ȳ - β₁x̄

### R² (Coefficient of Determination)
Proportion of variance in Y explained by the model: R² = 1 - SSresiduals/SStotal

R² = 0: Model explains nothing. R² = 1: Model explains all variance.

### Assumptions of Linear Regression
1. Linearity: True relationship is linear
2. Independence: Residuals are independent
3. Homoscedasticity: Constant variance of residuals
4. Normality: Residuals are normally distributed
