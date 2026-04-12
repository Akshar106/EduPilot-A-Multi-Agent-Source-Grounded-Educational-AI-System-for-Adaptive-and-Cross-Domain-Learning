# Statistics — Hypothesis Testing

## Introduction to Hypothesis Testing

Hypothesis testing is a formal statistical procedure for making decisions about population parameters using sample data. It answers: "Is this effect/difference real, or could it be due to random chance?"

### The Logic of Hypothesis Testing
We assume the null hypothesis is true and ask: "How likely is it to observe data this extreme or more extreme if the null is really true?" If this probability is very small, we have evidence against the null hypothesis.

## Setting Up Hypotheses

### Null Hypothesis (H₀)
The null hypothesis is the default assumption — typically that there is no effect, no difference, or no relationship. We try to find evidence against H₀.

Examples:
- H₀: μ = 100 (population mean equals 100)
- H₀: μ₁ = μ₂ (two population means are equal)
- H₀: There is no association between two categorical variables

### Alternative Hypothesis (H₁ or Hₐ)
What we want to test for — the claim that there IS an effect.

Types:
- **Two-tailed**: H₁: μ ≠ 100 (mean is different from 100, either direction)
- **Right-tailed**: H₁: μ > 100 (mean is greater than 100)
- **Left-tailed**: H₁: μ < 100 (mean is less than 100)

## The P-Value

### Definition
The **p-value** is the probability of observing a test statistic as extreme as (or more extreme than) the one calculated from the sample data, **assuming the null hypothesis is true**.

Formally: p = P(test statistic ≥ observed value | H₀ is true)

**Interpretation:**
- A small p-value (e.g., p = 0.02) means: if H₀ were true, there is only a 2% chance of seeing data this extreme. This is evidence against H₀.
- A large p-value (e.g., p = 0.45) means: even if H₀ were true, we would often see data this extreme. No compelling evidence against H₀.

### Significance Level (α)
The threshold below which we reject H₀. Commonly used values:
- α = 0.05 (5%) — most common in social sciences
- α = 0.01 (1%) — more stringent, medical research
- α = 0.001 (0.1%) — very stringent

### Decision Rule
- If p-value ≤ α → **Reject H₀** (result is "statistically significant")
- If p-value > α → **Fail to reject H₀** (result is "not statistically significant")

**Important:** "Fail to reject H₀" does NOT mean H₀ is true — it means we don't have enough evidence to reject it.

### Common Misconceptions About P-Values
1. **Wrong**: "The p-value is the probability that H₀ is true."  
   **Right**: It is the probability of observing this data (or more extreme) if H₀ were true.

2. **Wrong**: "p > 0.05 means there is no effect."  
   **Right**: We failed to find statistically significant evidence of an effect.

3. **Wrong**: "A small p-value means the effect is large."  
   **Right**: Statistical significance ≠ practical significance. With large samples, tiny effects can be statistically significant.

## Type I and Type II Errors

|  | H₀ is actually TRUE | H₀ is actually FALSE |
|--|---|---|
| **Reject H₀** | Type I Error (False Positive), probability = α | Correct (True Positive), probability = Power |
| **Fail to Reject H₀** | Correct (True Negative), probability = 1-α | Type II Error (False Negative), probability = β |

- **Type I Error (α)**: Rejecting a true H₀. Controlled by choosing α.
- **Type II Error (β)**: Failing to reject a false H₀.
- **Power** = 1 - β = probability of correctly rejecting a false H₀.

Power increases with: larger sample size, larger true effect size, larger α, smaller variability.

## The z-Test

Used when the population standard deviation σ is known and the sample is large.

**Test statistic:** z = (x̄ - μ₀) / (σ/√n)

Compare z to critical values from the standard normal distribution:
- Two-tailed at α = 0.05: reject if |z| > 1.96
- Right-tailed at α = 0.05: reject if z > 1.645
- Left-tailed at α = 0.05: reject if z < -1.645

## The t-Test

Used when σ is unknown (usual case) and/or sample size is small.

**Test statistic:** t = (x̄ - μ₀) / (s/√n)

Follows the t-distribution with df = n - 1.

### One-Sample t-Test
Tests whether a sample mean differs from a hypothesized value μ₀.

**Example:** A professor claims the average exam score is 75. We sample 25 students, get x̄ = 79, s = 10.
- t = (79 - 75) / (10/√25) = 4/2 = 2.0
- df = 24; critical t at α = 0.05 (two-tailed) ≈ 2.064
- Since |2.0| < 2.064, we fail to reject H₀ at the 5% level.

### Independent Samples t-Test (Two-Sample)
Compares means of two independent groups.

**H₀:** μ₁ = μ₂ (or equivalently, μ₁ - μ₂ = 0)

**Test statistic:**
t = (x̄₁ - x̄₂) / SE(x̄₁ - x̄₂)

Where SE = √(s₁²/n₁ + s₂²/n₂) (assuming unequal variances — Welch's t-test)

### Paired t-Test
Used when observations come in natural pairs (e.g., before/after measurements on the same subjects). More powerful than independent samples t-test when applicable.

**Procedure:** Compute differences dᵢ = xᵢ₂ - xᵢ₁, then do one-sample t-test on d.

## Chi-Square Tests

### Goodness-of-Fit Test
Tests whether an observed categorical distribution matches an expected distribution.

**H₀:** The data follows the expected distribution.
**Test statistic:** χ² = Σ[(Oᵢ - Eᵢ)² / Eᵢ]

Where Oᵢ = observed frequency, Eᵢ = expected frequency.
Degrees of freedom = number of categories - 1.

### Test of Independence (Contingency Table)
Tests whether two categorical variables are independent.

**H₀:** The two variables are independent.
**Test statistic:** Same χ² formula, applied to a 2D contingency table.
df = (rows - 1) × (columns - 1).

If χ² is large → p-value is small → evidence against independence.

## ANOVA (Analysis of Variance)

ANOVA tests whether the means of three or more groups are equal.

**Why not just do multiple t-tests?** Running multiple comparisons inflates the overall Type I error rate (multiple comparisons problem). If you do 20 t-tests at α = 0.05, you'd expect ~1 false positive even if there's no real effect.

**One-Way ANOVA**
- H₀: μ₁ = μ₂ = μ₃ = … = μₖ (all group means are equal)
- H₁: At least one group mean differs

**How it works:** Partitions total variance into:
- **Between-group variance** (SSB): Due to differences between group means
- **Within-group variance** (SSW): Due to random variation within groups

**F-statistic = MSB / MSW = (SSB/df_between) / (SSW/df_within)**

If F is large → between-group differences are large relative to within-group noise → evidence against H₀.

**Post-hoc tests:** If ANOVA rejects H₀, use post-hoc tests (Tukey HSD, Bonferroni) to find which specific pairs of groups differ.

## Effect Size

Statistical significance tells us "is the effect real?" but NOT "is the effect meaningful?"

### Cohen's d (for t-tests)
d = (x̄₁ - x̄₂) / s_pooled

Interpretation:
- d = 0.2: Small effect
- d = 0.5: Medium effect
- d = 0.8: Large effect

### Eta-Squared (η²) for ANOVA
η² = SSB / SST: proportion of variance explained by group membership.

### Practical vs. Statistical Significance
- **Statistical significance**: The effect is unlikely to be due to chance (p < α)
- **Practical significance**: The effect is large enough to matter in the real world

Example: With n = 10,000, even a difference of 0.1 IQ points can be statistically significant (p < 0.05), but this is practically meaningless.

## Multiple Testing Problem

When testing many hypotheses simultaneously, the chance of at least one false positive grows rapidly.

With k independent tests at α = 0.05:
P(at least one false positive) = 1 - (1 - 0.05)^k

For k = 20: ~64% chance of at least one false positive!

### Bonferroni Correction
Adjust significance threshold: α_adjusted = α / k

For α = 0.05 and 20 tests: α_adjusted = 0.0025

**Trade-off:** Very conservative — reduces Type I error but increases Type II error (misses real effects).

### False Discovery Rate (FDR) — Benjamini-Hochberg
Controls the expected proportion of false positives among all rejected tests. Less conservative than Bonferroni. Widely used in genomics and large-scale testing.

## Non-Parametric Tests

Used when data violates the assumptions of parametric tests (normality, equal variances).

| Parametric Test | Non-Parametric Equivalent |
|---|---|
| One-sample t-test | Wilcoxon Signed-Rank Test |
| Two-sample t-test | Mann-Whitney U Test |
| Paired t-test | Wilcoxon Signed-Rank Test |
| One-way ANOVA | Kruskal-Wallis Test |
| Pearson correlation | Spearman Rank Correlation |

Non-parametric tests are based on ranks rather than raw values. Less powerful than parametric tests when assumptions are met, but more robust when they're not.

## Bootstrap Methods

The **bootstrap** is a resampling technique for estimating the sampling distribution of almost any statistic without relying on distributional assumptions.

**Procedure:**
1. From your sample of n observations, draw n observations **with replacement** (this is the bootstrap sample)
2. Compute the statistic of interest (mean, median, R², etc.) for this bootstrap sample
3. Repeat steps 1-2 many times (typically 1,000–10,000)
4. The distribution of bootstrap statistics approximates the sampling distribution

**Applications:**
- Bootstrap confidence intervals (percentile method: take the 2.5th and 97.5th percentiles)
- Hypothesis testing
- Estimating standard errors for complex statistics
