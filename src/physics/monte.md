# **Monte Carlo Sampling for Uncertainty Estimation**
## **1. Mathematical Foundations of Monte Carlo Sampling**
### **1.1 Definition**
Monte Carlo sampling is a computational technique that uses repeated, independent random sampling from a specified probability distribution to estimate a target mathematical quantity (e.g an expected value, an integral, or a probability)

*Mathematically*

Given a random variable $\(X\)$ with a probability distribution $p(x)$ (or a function $\(f(x)\)$ whose expected value or integral we want to compute), the goal is to estimate the expected value:

$$\theta =\mathbb{E}[f(X)]=\int f(x)p(x)\,dx$$

1. Draw Samples: Generate a set of $"N"$ independent and identically distributed (i.i.d.) random samples $\(x_1, x_2, \dots, x_N\)$ from the target probability distribution $\(p(x)\)$.
2. Evaluate the Function: Compute the function value $\(f(x_i)\)$ for each sampled point $(x_{i})$.
3. Compute the Estimator: Approximate the expected value $(\theta)$ using the sample mean

$$\hat\theta_{N}=\frac{1}{N}\sum_{i=1}^{N}f(x_{i})$$


*Key Properties*
- **Law of Large Numbers**: As the number of samples approaches to infinity $(N \to \infty)$, the sample mean $\hat\theta_{N}$ converges almost surely to the true expected value $\theta$.

$$\lim _{N\rightarrow \infty }\frac{1}{N}\sum _{i=1}^{N}X_{i}=\mathbb{E}[X]$$

- **Central Limit Theorem**: The error of the estimate scales on the order of $(\frac{1}{\sqrt{N}})$, meaning the convergence rate is independent of the dimension of the domain (which helps avoid the curse of dimensionality compared to traditional grid-based numerical integration).

## **1.2 Monte Carlo Sampling Application**
In the standard neural network, a layer computes a function $f(x, \mathbf{W})$ with fixed weights $\mathbf{W}$. In a true Bayesian neural network, the weights are not fixed numbers. Instead, they follow a posterior probability distribution $p(\mathbf{W}|X, Y)$. To make a new prediction $(y^{\*})$ for a new input $(x^{\*})$, we mathematically must compute an expected value (an integral or entire posterior distribution) for over all possible weight given training data $(X,Y)$: 

$$p(y^{\*}|x^{\*},X,Y)=\int p(y^{\*}|x^{\*},W),p(W|X,Y),dW$$

This integral is analytically intractable (impossible to calculate exactly) because there are infinitely many combinations of weights.

A neural network contains millions of parameters, hence, this integral is infinitely complex. Monte Carlo sampling approximates this continuous integral by drawing $"N"$ discrete random weight configurations $(\hat{W}_i)$ from the distribution and by calculating a simple arithmetic mean: 

$$\int p(y^{\*}\mid x^{\*},W)\,p(W\mid X,Y)\,dW\approx \frac{1}{N}\sum _{i=1}^{N}p(y^{\*}\mid x^{\*},\widehat{W}_{i})$$

# **Monte Carlo Works in Data**
## **2. Estimating Uncertainty (Variance) from N-Sample Outputs**
When you run a stochastic model (like Monte Carlo Dropout) $"N"$ times for the exact same input, you receive $"N"$ distinct output vectors: $\({y_1, y_2, \dots, y_N}\)$. We use these samples to calculate two types of uncertainty: Epistemic (model uncertainty) and Aleatoric (inherent data noise). Here is how you mathematically compute the final prediction and its corresponding uncertainty. 

## **2.1 The Expected Prediction (First Moment)**
The final stable prediction is the Monte Carlo Sample Mean $\mu$, which represents the center of mass of your predictions: 

$$\mu =\frac{1}{N}\sum _{i=1}^{N}y_{i}$$

## **2.2 The Predictive Uncertainty (Second Moment / Variance)**
Uncertainty is quantified by calculating the Sample Variance $\sigma ^{2}$ across the $"N"$ outputs. Variance measures the average squared deviation of each individual random trial from the sample mean: 

$$\sigma ^{2}=\frac{1}{N-1}\sum _{i=1}^{N}(y_{i}-\mu )^{2}$$

## **2.3 Interpreting the Variance Value**
- **Low Variance $(\sigma^2 \to 0)$**: The $"N"$ different random configurations of your network all arrived at nearly identical conclusions. The model is highly confident in its prediction. 

- **High Variance $(\sigma^2 \gg 0\)$**: The random configurations generated widely different outputs. This indicates that the input lies in a region of the data space where the model's parameters are unconstrained (high epistemic uncertainty). The model is guessing. 

## **2.4 Step-by-Step Mathematical Flow Through the Network**
To document how this maps onto neural network architecture (e.g 4 hidden layers, 64 neurons each), the system flows sequentially through three mathematical layers: 

```
[ Input Tensor ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. THE MICRO LAYER: Bernoulli Distribution             │
│    - Applied to each of the 256 individual neurons.    │
│    - M_i ~ Bernoulli(1 - p)                            │
│    - Yields a hard binary mask element: 0 or 1.        │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. THE MACRO LAYER: Binomial Distribution              │
│    - Applied to each of the 4 individual layers.       │
│    - K ~ Binomial(n=64, p_keep=1-p)                    │
│    - Governs the total count of active hidden nodes.   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 3. THE INFRASTRUCTURE LAYER: Monte Carlo Loop          │
│    - Evaluates N forward passes over time.             │
│    - Computes Sample Mean (μ) and Sample Variance (σ²) │
└────────────────────────────────────────────────────────┘
       │
       ▼
[ Final Prediction + Uncertainty Map ]
```
1. Bernoulli Level (Per Neuron): A single node outputs $x_i \cdot M_i \cdot \frac{1}{1-p}$, where $M_i \in \{0, 1\}$. 
2. Binomial Level (Per Layer): The layer acts as a pool of 64 independent Bernoulli trials. The probability that exactly $"k"$ nodes stay active during a pass is dictated by $P(K=k) = \binom{64}{k}(1-p)^k p^{64-k}$. Monte Carlo Level (The System): The loops generate $"N"$ unique matrix permutations of the network. The final script reduces these dimensions using the variance formula to yield a tangible uncertainty map.


