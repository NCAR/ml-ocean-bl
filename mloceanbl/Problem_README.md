# Estimation of Mixed Layer Depth 

## Problem Formulation
We consider the estimation of the mixed layer depth on a $1/2^\circ$ degree grid as a function of sea surface salinity, temperature, and height anomaly. We will use sparse, informative mixed layer depth information from Argo profiles as data to inform the model. Let $S(z,t), T(z,t), H(z,t)$ stand for salinity, temperature, and height respectively at spatial location $z$ and time $t$. Let $D(z,t)$ be the mixed layer depth at $(z,t)$ and let $d_j(t)$ refer to all argo mixed layer data at time $t$ (the location of these data points, say $z_j$ for $j = 1...n$ do not necessary coincide with any elements of $z$.

We aim to learn a functional relationship between $D$ and $S, T$, and $H$. $d(t)$ and $D(z,t)$ are then related by a linear transformation that projects the $z$ grid onto the locations of the Argo data points $z_j$.  

\begin{equation*}
    \begin{aligned}
    D &= f(S, T, H) + \sigma, \quad \sigma \sim \mathcal{N}(0, \pmb{Q}) \\
    d(t) &= \pmb{L}(t) D(z,t) + \epsilon(t), \; \; \epsilon(t) \sim \mathcal{N}(0, \pmb{L}(t)\pmb{Q}\pmb{L}(t) + \pmb{V}(t)) \\
    \end{aligned}
\end{equation*}

## Quantifying Uncertainty
Along with learning the functional relationship $f$, we are also ultimately interested in estimating the covariance $\pmb{Q}$. The linear maps $\pmb{L}(t)$ and $\pmb{V}(t)$ are known as they can be calculuated via Gaussian Process Regression. $\pmb{V}(t)$ represents the uncertainty arising from the spatial interpolation process. The joint likelihood for $D$ and $\pmb{Q}$ given $d(t)$ is 
\begin{equation}
p(d(t) | D, \pmb{Q} ) \propto \exp{\biggl(-\frac{1}{2}(d - \pmb{L}D)^T(\pmb{L}\pmb{Q}\pmb{L} + \pmb{V})^{-1}(d-\pmb{L}D)\biggr)}
\end{equation}
The prior for $D$ is given by
\begin{equation}
p(D|\pmb{Q}) \propto \exp{\bigl(-\frac{1}{2}\left(D - f(S,T,H)\right)^T\pmb{Q}^{-1}\left(D - f(S,T,H)\right) \bigr)}
\end{equation}
The posterior probability distribution is given by
\begin{equation}
p(D, \pmb{Q} | d) \propto p(d|D, \pmb{Q})p(D|\pmb{Q})p(\pmb{Q})
\end{equation}

## Learning functional relationship
It is simplest to learn the weights of $f$ by first minimizing a negative log likelihood that does not account for the noise $\pmb{Q}$.
\begin{equation}
\hat{f} = \underset{f}{\operatorname{argmin}} \biggl( \frac{1}{2}(d - \pmb{L}f(S,T,H))^T(\pmb{V})^{-1}(d-\pmb{L}f(S,T,H))\biggr)
\end{equation}

However tensorflow-probability has built in methods to find $f$ and monte-carlo sample to estimate $\pmb{Q}$ simultaneously. Cost may be prohibitive, to be investigated.

### Estimating p(Q)

A prior for $\pmb{Q}$ can be specified but this is likely to be less informative than if we estimate the noise during the learning of $f$. Techniques depend on the specific implementation. If $f$ is linear, than an explicit relationship can be derived given assumptions about the uncertainties in $S,T$ and $H$. If $f$ is non-linear, than we may have to precede using Monte Carlo sampling of weights, dropout, k-fold witholding. 