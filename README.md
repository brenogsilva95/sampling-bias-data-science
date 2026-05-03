# sampling-bias-data-science-animation
Visual explanation of sampling bias in data science using simulated data and animated storytelling with Python.

# Sampling Bias in Data Science: An Animated Explanation

Post: https://www.linkedin.com/posts/brenogdsilva_nem-todo-problema-em-ci%C3%AAncia-de-dados-come%C3%A7a-activity-7452357371978407937-a_Vz?utm_source=share&utm_medium=member_desktop&rcm=ACoAADzYEn0BidO853LZcLlVhNbNq-HOPuFNURk

## Introduction

In data science, poor model performance is not always caused by the algorithm itself. In many real-world applications, the problem begins earlier: in the data collection process.

Sampling bias occurs when the observed sample is not representative of the target population. As a consequence, models may learn patterns that reflect the data acquisition mechanism rather than the real structure of the population. This issue is closely related to sample selection bias and dataset shift, both widely discussed in machine learning and statistical learning literature.

Zadrozny (2004) formalized sample selection bias in machine learning and showed that learning and evaluation can be affected when the observed sample differs systematically from the target population. Quiñonero-Candela et al. (2009) discuss dataset shift as a common problem in predictive modeling, especially when the joint distribution of inputs and outputs differs between training and deployment. Lazer et al. (2014) also highlight that large datasets can still lead to misleading conclusions when the data-generating process is biased.

This repository provides a visual and educational simulation showing how a seemingly balanced population can become biased after a selective observation process. The goal is to communicate a central idea in statistics and data science: a good model cannot fully compensate for biased data.

## Problem Context

Suppose there is a real population composed of two groups, Group A and Group B. In the full population, both groups are balanced. However, only a restricted region of the feature space is observed.

This selective observation mechanism changes the group composition of the sample.

As a result, conclusions based only on the observed sample may not represent the true population.

## Statistical Framework

Let the population distribution be represented by:

$$
P(X, Y, G)
$$

where:

- $X$ and $Y$ are observed features
- $G$ is the group membership variable

The target population proportion for group $g$ is:

$$
\pi_g = P(G = g)
$$

In practice, we often observe data only when a selection indicator $S$ is equal to 1:

$$
S =
\begin{cases}
1, & \text{if the observation is included in the sample} \\
0, & \text{otherwise}
\end{cases}
$$

The observed sample distribution is therefore:

$$
P(X, Y, G \mid S = 1)
$$

Sampling bias occurs when:

$$
P(G = g \mid S = 1) \neq P(G = g)
$$

or, more generally:

$$
P(X, Y, G \mid S = 1) \neq P(X, Y, G)
$$

This means that the sample is not a faithful representation of the target population.

## Simulation Design

The simulation generates two groups from different bivariate distributions.

For Group A:

$$
X_A \sim N(4.2, 1.2^2)
$$

$$
Y_A \sim N(5.0, 1.4^2)
$$

For Group B:

$$
X_B \sim N(6.5, 1.3^2)
$$

$$
Y_B \sim N(4.2, 1.2^2)
$$

The population is balanced:

$$
P(G = A) = P(G = B) = 0.5
$$

However, the observed sample is defined by a selection window:

$$
S = 1 \quad \text{if} \quad X > 5.2,\ 3.5 < Y < 6.8
$$

This creates a biased sample because the probability of being observed depends on the feature values:

$$
P(S = 1 \mid X, Y, G) \neq P(S = 1)
$$

## Methodology

The project uses simulated data to illustrate the selection process step by step.

First, the full population is generated with two groups. Then, a selection window is imposed on the feature space. Observations inside this window are treated as the available sample, while observations outside the window are faded in the visualization.

The animation is structured in four stages:

1. Display of the full population
2. Appearance of the observation window
3. Highlighting of the biased sample
4. Comparison between population and sample proportions

The visualization was created using Python, NumPy, Matplotlib, and `FuncAnimation`.

The code used in this repository is based on the animation script provided in the attached file. :contentReference[oaicite:0]{index=0}

## Results

The simulation shows that the full population starts balanced between Group A and Group B.

However, after applying the observation window, the sample proportions change. This demonstrates that the data available for modeling may no longer reflect the population structure.

The key result is not a predictive model, but a statistical insight:

$$
\text{Biased data} \Rightarrow \text{biased conclusions}
$$

Even before fitting a machine learning algorithm, the data collection process can distort the problem.

## Data Science Interpretation

In applied data science, sampling bias can affect:

- model training
- performance evaluation
- fairness analysis
- business decisions
- generalization to production environments

If the model is trained on a biased sample, it may perform well on similar biased data but fail when applied to the real target population.

This is closely related to dataset shift, where:

$$
P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)
$$

In this project, the observed sample plays the role of a biased training dataset, while the full population represents the real-world target distribution.
