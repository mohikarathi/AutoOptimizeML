# Mathematical Optimization Formulation

AutoOptimizeML formulates configuration discovery as a **constrained multi-objective optimization problem**.

---

## 1. Problem Formulation

Let \( \mathcal{X} \) denote the bounded space of feasible execution configurations:

\[
x = (\text{backend}, \text{precision}, \text{batch\_size}, \text{workers}, \text{compile\_graph}, \text{native\_accel}) \in \mathcal{X}
\]

The optimization goal is:

\[
\begin{aligned}
\max_{x \in \mathcal{X}} \quad & f(x) = \text{Throughput}(x) \quad (\text{or } \min_{x \in \mathcal{X}} \text{Latency}(x)) \\
\text{subject to} \quad & g_1(x) = \text{Latency}(x) - L_{\max} \le 0 \\
& g_2(x) = A_{\min} - \text{Accuracy}(x) \le 0 \\
& g_3(x) = \text{Memory}(x) - M_{\max} \le 0
\end{aligned}
\]

where:
* \( L_{\max} \) is the maximum acceptable latency in milliseconds.
* \( A_{\min} \) is the minimum acceptable model accuracy threshold (e.g. 0.90).
* \( M_{\max} \) is the maximum RAM/VRAM memory budget in megabytes.

---

## 2. Constraint Penalty & Utility Scoring

For unconstrained ranking among valid candidates:
* **Throughput Objective**: \( S(x) = \text{Throughput}(x) \) (samples/sec)
* **Latency Objective**: \( S(x) = \frac{1000}{\max(10^{-4}, \text{Latency}(x))} \)
* **Balanced Pareto Utility**:
  \[
  S(x) = 0.6 \cdot \text{Throughput}(x) + 0.4 \cdot \left( \frac{1000}{\text{Latency}(x)} \right) \cdot \text{Accuracy}(x)
  \]

If any hard constraint is violated (\( g_i(x) > 0 \)), the candidate is marked `REJECTED` and its score is penalized to \( S(x) = -10^6 \).

---

## 3. Bayesian Optimization via Gaussian Process Surrogates

To avoid evaluating inefficient configurations, AutoOptimizeML models the objective function \( f(x) \) with a Gaussian Process surrogate:

\[
f(x) \sim \mathcal{GP}\left( \mu(x), k(x, x') \right)
\]

where \( k(x, x') \) is the Matérn \( 5/2 \) covariance kernel:

\[
k_{\text{Matérn5/2}}(r) = \sigma^2 \left( 1 + \sqrt{5}\frac{r}{\ell} + \frac{5r^2}{3\ell^2} \right) \exp\left(-\sqrt{5}\frac{r}{\ell}\right)
\]

### Acquisition Function: Expected Improvement (EI)

Candidate selection at iteration \( t+1 \) maximizes Expected Improvement over the current best observed score \( y^+ = \max_{i \le t} y_i \):

\[
\text{EI}(x) = \mathbb{E}\left[ \max(0, f(x) - y^+ - \xi) \right] = (\mu(x) - y^+ - \xi)\Phi(Z) + \sigma(x)\phi(Z)
\]

where:
\[
Z = \begin{cases} 
\frac{\mu(x) - y^+ - \xi}{\sigma(x)} & \text{if } \sigma(x) > 0 \\ 
0 & \text{if } \sigma(x) = 0 
\end{cases}
\]
and \( \Phi(\cdot), \phi(\cdot) \) are the standard normal cumulative distribution and probability density functions, respectively.
