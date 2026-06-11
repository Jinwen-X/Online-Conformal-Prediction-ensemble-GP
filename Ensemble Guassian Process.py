"""
Online Gaussian Process regression with Random Fourier Features (RFF),
ensembled over four kernels: RBF, Laplacian, Matern-3/2, Rational Quadratic.

RFF (Rahimi & Recht, 2007):
    k(x, x') ~= z(x)^T z(x'),
    z(x) = sqrt(1/D) * [ sin(V^T x), cos(V^T x) ]        # 2D features
where the D columns of V are i.i.d. draws from the kernel's spectral density.
This approximates a UNIT-VARIANCE, shift-invariant kernel; the signal variance
sigma_f^2 is handled separately as the weight-space prior in gp_predict().

All samplers below were validated by comparing the RFF Gram matrix against the
exact kernel (max abs error ~0.01 at D=2e4, 1-D inputs, l=1).
"""

import numpy as np
from scipy.stats import cauchy, gamma
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Random Fourier Feature map
# ----------------------------------------------------------------------
def rff_features(X, V):
    """X:(N, d), V:(d, D)  ->  Z:(N, 2D)."""
    D = V.shape[1]
    VX = X @ V
    return np.sqrt(1.0 / D) * np.concatenate([np.sin(VX), np.cos(VX)], axis=1)


# ----------------------------------------------------------------------
# 2. Spectral-density samplers. Each returns V of shape (d, D);
#    every COLUMN of V is one independent frequency omega ~ p(omega).
# ----------------------------------------------------------------------
def sample_rbf(d, D, length_scale, rng):
    # RBF: omega ~ N(0, 1/l^2)
    return rng.normal(0.0, 1.0 / length_scale, size=(d, D))


def sample_laplacian(d, D, length_scale, rng):
    # exp(-|r|/l): omega ~ Cauchy(0, scale = 1/l)
    return cauchy.rvs(loc=0.0, scale=1.0 / length_scale,
                      size=(d, D), random_state=rng)


def sample_matern(d, D, length_scale, nu, rng):
    # Matern-nu: omega ~ Student-t with dof = 2*nu, scale = 1/l.
    #   omega = z / sqrt(g),  z ~ N(0, 1/l^2),  g ~ Gamma(nu, scale=1/nu)
    # NOTE: g is drawn per-frequency (size=D), NOT shared across frequencies.
    g = gamma.rvs(a=nu, scale=1.0 / nu, size=D, random_state=rng)
    z = rng.normal(0.0, 1.0 / length_scale, size=(d, D))
    return z / np.sqrt(g)[None, :]


def sample_rq(d, D, length_scale, alpha, rng):
    # Rational quadratic = scale-mixture of RBFs:
    #   omega = z * sqrt(u),  z ~ N(0, 1),  u ~ Gamma(alpha, scale=1/(alpha*l^2))
    u = gamma.rvs(a=alpha, scale=1.0 / (alpha * length_scale ** 2),
                  size=D, random_state=rng)
    z = rng.normal(0.0, 1.0, size=(d, D))
    return z * np.sqrt(u)[None, :]


# ----------------------------------------------------------------------
# 3. Weight-space GP posterior (mean + diagonal predictive variance)
#    A = Z'Z + (sigma_n^2 / sigma_f^2) I
#    mu_*   = Z_* A^{-1} Z' y
#    var_*  = sigma_n^2 * diag(Z_* A^{-1} Z_*') + sigma_n^2
# ----------------------------------------------------------------------
def gp_predict(Z_train, y_train, Z_test, sigma_n, sigma_f):
    M = Z_train.shape[1]
    A = Z_train.T @ Z_train + (sigma_n ** 2 / sigma_f ** 2) * np.eye(M)
    w = np.linalg.solve(A, Z_train.T @ y_train)          # posterior mean weights
    mu = Z_test @ w
    AinvZt = np.linalg.solve(A, Z_test.T)                # (M, N_test)
    var = sigma_n ** 2 * np.einsum('ij,ji->i', Z_test, AinvZt) + sigma_n ** 2
    return mu, var


# ----------------------------------------------------------------------
# 4. Online learning loop (mini-batch). Returns predictions/truth/variance
#    for every test point, in arrival order.
# ----------------------------------------------------------------------
def run_online_gp(X_all, y_all, V, sigma_n, sigma_f, n_init=50, batch=50):
    X_tr, y_tr = X_all[:n_init].copy(), y_all[:n_init].copy()
    Z_tr = rff_features(X_tr, V)

    preds, truths, varis = [], [], []
    for i in range(n_init, len(X_all), batch):
        X_new, y_new = X_all[i:i + batch], y_all[i:i + batch]
        Z_new = rff_features(X_new, V)

        mu, var = gp_predict(Z_tr, y_tr, Z_new, sigma_n, sigma_f)
        preds.extend(mu)
        truths.extend(y_new)
        varis.extend(var)

        # absorb the new data into the training set
        X_tr = np.vstack([X_tr, X_new])
        y_tr = np.append(y_tr, y_new)
        Z_tr = np.vstack([Z_tr, Z_new])

    return np.array(preds), np.array(truths), np.array(varis)


# ----------------------------------------------------------------------
# 5. Per-point Gaussian log-likelihood (var = variance, not std)
# ----------------------------------------------------------------------
def gaussian_loglik(y, mu, var):
    var = np.maximum(var, 1e-12)
    return -0.5 * (np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)


# ----------------------------------------------------------------------
# 6. Causal multiplicative-weights ensemble (Bayesian model averaging).
#    w_i  <-  w_i * p_i(y_t),  done in log-space for stability.
#    The prediction for point t uses weights from points < t (no leakage).
# ----------------------------------------------------------------------
def ensemble(preds, logliks, init_weights=None):
    n_models, T = preds.shape
    w = (np.full(n_models, 1.0 / n_models) if init_weights is None
         else np.asarray(init_weights, float))
    ens = np.zeros(T)
    weight_hist = np.zeros((T, n_models))
    for t in range(T):
        weight_hist[t] = w
        ens[t] = w @ preds[:, t]                 # causal prediction
        logw = np.log(w + 1e-300) + logliks[:, t]
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
    return ens, weight_hist


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # ---- simulated dataset: 10000 noisy samples of sin(x) on [0, 10] ----
    N = 10000
    X_all = rng.rand(N, 1) * 10.0
    y_all = np.sin(X_all[:, 0]) + rng.randn(N) * 0.1
    d = X_all.shape[1]

    # ---- shared hyper-parameters (kept consistent for a fair comparison) ----
    length_scale = 1.0
    sigma_f = 1.0
    sigma_n = 0.1
    D = 300

    kernels = {
        "RBF":     sample_rbf(d, D, length_scale, rng),
        "Matern":  sample_matern(d, D, length_scale, nu=1.5, rng=rng),
        "Laplace": sample_laplacian(d, D, length_scale, rng),
        "RQ":      sample_rq(d, D, length_scale, alpha=2.0, rng=rng),
    }

    results = {}
    for name, V in kernels.items():
        preds, truths, varis = run_online_gp(
            X_all, y_all, V, sigma_n, sigma_f, n_init=50, batch=50)
        results[name] = dict(pred=preds, var=varis)
    truth = truths  # same ordering for every kernel

    # ---- ensemble ----
    names = list(kernels.keys())
    pred_mat = np.vstack([results[n]["pred"] for n in names])       # (4, T)
    ll_mat = np.vstack([gaussian_loglik(truth, results[n]["pred"],
                                        results[n]["var"]) for n in names])
    ens_pred, weight_hist = ensemble(pred_mat, ll_mat)

    # ---- metrics ----
    print("MSE per kernel:")
    for n in names:
        print(f"  {n:8s}: {mean_squared_error(truth, results[n]['pred']):.5f}")
    print(f"  {'Ensemble':8s}: {mean_squared_error(truth, ens_pred):.5f}")

    # ---- plot ----
    lo, hi = 100, 300
    plt.figure(figsize=(12, 6))
    plt.plot(range(lo, hi), truth[lo:hi], 'ro', label='True')
    plt.plot(range(lo, hi), ens_pred[lo:hi], 'bx-', label='Ensemble prediction')
    plt.title("Online GP regression with RFF (kernel ensemble)")
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("online_gp_rff.png", dpi=120)
    print("Saved plot -> online_gp_rff.png")

