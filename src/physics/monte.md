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

# **Monte Carlo Sampling in Dataset**
## **2. Estimating Uncertainty (Variance) from N-Sample Outputs**
When we run a stochastic model (like Monte Carlo Dropout) $"N"$ times for the exact same input, we receive $"N"$ distinct output vectors: $\({y_1, y_2, \dots, y_N}\)$. We use these samples to calculate two types of uncertainty: Epistemic (model uncertainty) and Aleatoric (inherent data noise). Here's how we mathematically compute the final prediction and its corresponding uncertainty. 

## **2.1 The Expected Prediction (First Moment)**
The final stable prediction is the Monte Carlo Sample Mean $\mu$, which represents the center of mass of your predictions: 

$$\mu =\frac{1}{N}\sum _{i=1}^{N}y_{i}$$

## **2.2 The Predictive Uncertainty (Second Moment / Variance)**
Uncertainty is quantified by calculating the Sample Variance $\sigma ^{2}$ across the $"N"$ outputs. Variance measures the average squared deviation of each individual random trial from the sample mean: 

$$\sigma ^{2}=\frac{1}{N-1}\sum _{i=1}^{N}(y_{i}-\mu )^{2}$$

## **2.3 Interpreting the Variance Value**
- **Low Variance $(\sigma^2 \to 0)$**: The $"N"$ different random configurations of the network all arrived at nearly identical conclusions. The model is highly confident in its prediction. 

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
1. **Bernoulli Level (Per Neuron)**: A single node outputs $x_i\cdot M_i\cdot\frac{1}{1-p}$, where $M_i\in\{0, 1\}$. 
2. **Binomial Level (Per Layer)**: The layer acts as a pool of 64 independent Bernoulli trials. The probability that exactly $"k"$ nodes stay active during a pass is dictated by $P(K=k) = \binom{64}{k}(1-p)^k p^{64-k}$. Monte Carlo Level (The System): The loops generate $"N"$ unique matrix permutations of the network. The final script reduces these dimensions using the variance formula to compute a tangible uncertainty map.

# **Example: Analytical Method for 50 sample**

To calculate the predictive uncertainty (variance) from the 50 outputs of the 4-layer network, we treat the outputs as 50 independent random samples. If the network is doing a regression task (like predicting continuous values for an MRI translation or reconstruction), each output will be a tensor of numerical predictions. 

**Step 1: The Raw Data Array (The Collection)**
When we pass input through the 50 Monte Carlo passes, we save the raw outputs into a 2D matrix. Let's assume the network predicts an output vector of size D (e.g., D could be a flattened patch of pixels or a vector of continuous features). We collect these into a matrix $\mathbf{Y}$ of shape (50, D): 

$$\mathbf{Y} =
\begin{bmatrix}
y_{1,1} & y_{1,2} & \cdots & y_{1,D} \\
y_{2,1} & y_{2,2} & \cdots & y_{2,D} \\
\vdots  & \vdots  & \ddots & \vdots  \\
y_{50,1} & y_{50,2} & \cdots & y_{50,D}
\end{bmatrix}$$

Where $y_{t,j}$ is the prediction at the $j-th$ feature/pixel during the $t-th$ Monte Carlo pass. 

**Step 2: Compute the Sample Mean $(\mu)$**

Before finding the variance, we must find the center of mass (the mean) for every single feature dimension independently. We sum up the 50 samples along the columns and divide by 50. For any specific feature j:
$\mu_{j}=\frac{1}{50}\sum_{t=1}^{50}y_{t,j}$
This gives an **Expected Prediction Vector** $\boldsymbol{\mu} = \mu_1, \mu_2, \dots, \mu_D $. 

**Step 3: Compute the Sample Variance $(\sigma ^{2})$**

The variance represents the model's uncertainty. To compute it, we measure how far each of the 50 random predictions deviates from that mean vector $\mathbfit{\mu }$. For any specific feature $j$, the sample variance $\sigma _{j}^{2}$ is calculated as: 

$$\sigma _{j}^{2}=\frac{1}{50-1}\sum _{t=1}^{50}(y_{t,j}-\mu _{j})^{2}$$

Why (50 - 1)? We divide by 49 instead of 50 (Bessel’s Correction). This mathematically corrects for the bias introduced because we are estimating the true hidden population mean using our small sample of 50 runs. This yields an Uncertainty Vector $\boldsymbol{\sigma^2} = \sigma^2_1, \sigma^2_2, \dots, \sigma^2_D$. 

**Step 4: Concrete Numerical Example (For 1 Single Pixel/Feature)**
For a single pixel or feature $(j=1)$ across just 3 dummy passes to see the arithmetic: 
- Pass 1 output $(y_{1,1})$: 10.0
- Pass 2 output $(y_{2,1})$: 12.0
- Pass 3 output $(y_{3,1})$: 8.0 

**1. Calculate Mean $(\mu _{1})$**: 

$$\mu _{1}=\frac{10.0+12.0+8.0}{3}=\frac{30.0}{3}=\mathbf{10.0}$$

**2. Calculate Squared Deviations**:
- Pass 1: $(10.0 - 10.0)^2$ = 0.0
- Pass 2: $(12.0 - 10.0)^2$ = 4.0
- Pass 3: $(8.0 - 10.0)^2$ = 4.0

**3. Sum and Divide by (N-1)**: 

$$\sigma _{1}^{2}=\frac{0.0+4.0+4.0}{3-1}=\frac{8.0}{2}=\mathbf{4.0}$$

The final prediction for this feature is 10.0 with an uncertainty variance of 4.0 (or a standard deviation of $\sqrt{4} = 2.0$). 




