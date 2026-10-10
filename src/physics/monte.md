# **Monte Carlo Sampling for Uncertainty Estimation**
## **1. Mathematical Foundations of Monte Carlo Sampling**
### **1.1 Definition**
Monte Carlo sampling is a computational technique that uses repeated, independent random sampling from a specified probability distribution to estimate a target mathematical quantity (such as an expected value, an integral, or a probability)

*Mathematically*

Given a random variable $\(X\)$ with a probability distribution $\(p(x)\)$ (or a function $\(f(x)\)$ whose expected value or integral we want to compute), the goal is to estimate the expected value:

$$\theta =\mathbb{E}[f(X)]=\int f(x)p(x)\,dx$$

1. Draw Samples: Generate a set of $\(N\)$ independent and identically distributed (i.i.d.) random samples $\(x_1, x_2, \dots, x_N\)$ from the target probability distribution $\(p(x)\)$.
2. Evaluate the Function: Compute the function value $\(f(x_i)\)$ for each sampled point.
3. Compute the Estimator: Approximate the expected value $\(\theta \)$ using the sample mean

$$\hat\theta_{N}=\frac{1}{N}\sum_{i=1}^{N}f(x_{i})$$


*Key Properties*
- **Law of Large Numbers**: As the number of samples increase $\(N \to \infty\)$, the sample mean $\(\^{\theta }_{N}\)$ converges almost surely to the true expected value $\(\theta \)$.
$$\[\lim _{N\rightarrow \infty }\frac{1}{N}\sum _{i=1}^{N}X_{i}=\mathbb{E}[X]\]$$
- **Central Limit Theorem**: The error of the estimate scales on the order of $\(\frac{1}{\sqrt{N}}\)$, meaning the convergence rate is independent of the dimension of the domain (which helps avoid the curse of dimensionality compared to traditional grid-based numerical integration)


