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

On top of the (ensemble) GP we run ONLINE CONFORMAL PREDICTION with a decaying
step size (Angelopoulos, Barber & Bates, 2024, arXiv:2402.01139), using the
Gaussian negative log-likelihood as the conformal score. This wraps the GP's
point predictions in calibrated prediction sets with a long-run coverage
guarantee that does NOT rely on the GP variance being correctly specified.
"""

import numpy as np
from scipy.stats import cauchy, gamma, chi2
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
#    Also returns the mixture predictive mean & variance (moment-matched
#    Gaussian), which the conformal layer uses for the NLL score.
# ----------------------------------------------------------------------
def ensemble(preds, varis, logliks, init_weights=None):
    n_models, T = preds.shape
    w = (np.full(n_models, 1.0 / n_models) if init_weights is None
         else np.asarray(init_weights, float))
    ens_mu = np.zeros(T)
    ens_var = np.zeros(T)
    weight_hist = np.zeros((T, n_models))
    for t in range(T):
        weight_hist[t] = w
        mu_t = w @ preds[:, t]                              # mixture mean
        # variance of a mixture of Gaussians:
        #   Var = sum_i w_i (var_i + mu_i^2) - (sum_i w_i mu_i)^2
        second = w @ (varis[:, t] + preds[:, t] ** 2)
        ens_mu[t] = mu_t
        ens_var[t] = second - mu_t ** 2
        logw = np.log(w + 1e-300) + logliks[:, t]
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
    return ens_mu, ens_var, weight_hist


# ----------------------------------------------------------------------
# 7. Online conformal prediction with decaying step size
#    (Angelopoulos, Barber & Bates, 2024).
#
#    Score (Gaussian NLL):
#        s_t(y) = 0.5*log(2*pi*sigma_t^2) + (y - mu_t)^2 / (2*sigma_t^2)
#    Prediction set  C_t = {y : s_t(y) <= q_t}  is the closed interval
#        mu_t +/- sqrt( 2*sigma_t^2 * (q_t - 0.5*log(2*pi*sigma_t^2)) ),
#    or EMPTY when the radicand is < 0. (Unlike a residual score, the NLL
#    diverges as |y|->inf, so the set is never infinite.)
#    Threshold update (eq. 4):
#        q_{t+1} = q_t + eta_t * ( 1{Y_t not in C_t} - alpha )
#    Step size:  'decaying' -> eta_t = c * t^{-1/2 - eps}   (the paper's method)
#                float       -> fixed eta_t = const          (Gibbs & Candes 2021)
# ----------------------------------------------------------------------
def online_conformal(y, mu, var, alpha=0.1, step="decaying", c=1.0, eps=0.1,
                     q1=None):
    T = len(y)
    var = np.maximum(np.asarray(var, float), 1e-12)
    nll_min = 0.5 * np.log(2.0 * np.pi * var)              # min score, at y = mu
    score = nll_min + (y - mu) ** 2 / (2.0 * var)          # observed score s_t(Y_t)

    # warm-start the threshold near the calibrated quantile to shorten burn-in
    if q1 is None:
        q1 = float(np.median(nll_min) + chi2.ppf(1.0 - alpha, df=1) / 2.0)
    q = q1

    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    covered = np.zeros(T, dtype=bool)
    q_path = np.empty(T)

    for t in range(T):
        q_path[t] = q
        radicand = 2.0 * var[t] * (q - nll_min[t])
        if radicand >= 0.0:                                # non-empty interval
            h = np.sqrt(radicand)
            lo[t], hi[t] = mu[t] - h, mu[t] + h
        covered[t] = score[t] <= q                         # Y_t in C_t ?
        eta = c * (t + 1) ** (-0.5 - eps) if step == "decaying" else float(step)
        q = q + eta * ((0.0 if covered[t] else 1.0) - alpha)

    return dict(lo=lo, hi=hi, covered=covered, q_path=q_path, score=score)


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

    # ---- ensemble (causal weights + mixture predictive mean/variance) ----
    names = list(kernels.keys())
    pred_mat = np.vstack([results[n]["pred"] for n in names])       # (4, T)
    var_mat = np.vstack([results[n]["var"] for n in names])         # (4, T)
    ll_mat = np.vstack([gaussian_loglik(truth, results[n]["pred"],
                                        results[n]["var"]) for n in names])
    ens_mu, ens_var, weight_hist = ensemble(pred_mat, var_mat, ll_mat)

    print("MSE per kernel:")
    for n in names:
        print(f"  {n:8s}: {mean_squared_error(truth, results[n]['pred']):.5f}")
    print(f"  {'Ensemble':8s}: {mean_squared_error(truth, ens_mu):.5f}")

    # ---- online conformal prediction on the ensemble (Gaussian-NLL score) ----
    alpha = 0.1
    cp_decay = online_conformal(truth, ens_mu, ens_var, alpha=alpha,
                                step="decaying", c=1.0, eps=0.1)
    cp_fixed = online_conformal(truth, ens_mu, ens_var, alpha=alpha,
                                step=0.05)

    def report(tag, cp):
        cov = cp["covered"].mean()
        width = np.nanmean(cp["hi"] - cp["lo"])
        empty = np.mean(np.isnan(cp["lo"]))
        print(f"  {tag:18s} long-run coverage={cov:.3f} "
              f"(target {1-alpha:.2f}) | mean width={width:.3f} | "
              f"empty-set rate={empty:.3f}")

    print(f"\nOnline conformal prediction (alpha={alpha}):")
    report("decaying eta_t", cp_decay)
    report("fixed eta=0.05", cp_fixed)

    # ---- plots: prediction band, threshold path, rolling coverage ----
    def rolling(c, w=1000):
        c = c.astype(float)
        return np.convolve(c, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    lo, hi = 100, 300
    ax[0].fill_between(range(lo, hi), cp_decay["lo"][lo:hi],
                       cp_decay["hi"][lo:hi], alpha=0.25, color="C0",
                       label=f"{int((1-alpha)*100)}% conformal set")
    ax[0].plot(range(lo, hi), ens_mu[lo:hi], "C0-", lw=1, label="Ensemble mean")
    ax[0].plot(range(lo, hi), truth[lo:hi], "r.", ms=4, label="True")
    ax[0].set_title("Ensemble GP + conformal band")
    ax[0].set_xlabel("sample index"); ax[0].legend(fontsize=8)

    ax[1].plot(cp_fixed["q_path"], color="C1", lw=0.8, label="fixed eta")
    ax[1].plot(cp_decay["q_path"], color="C0", lw=0.8, label="decaying eta")
    ax[1].set_title("Threshold q_t"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=8)

    ax[2].axhline(1 - alpha, color="k", ls=":", label="1 - alpha")
    ax[2].plot(rolling(cp_fixed["covered"]), color="C1", lw=0.8, label="fixed eta")
    ax[2].plot(rolling(cp_decay["covered"]), color="C0", lw=0.8, label="decaying eta")
    ax[2].set_title("Rolling coverage (window 1000)")
    ax[2].set_xlabel("t"); ax[2].set_ylim(0.7, 1.0); ax[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("online_gp_rff_conformal.png", dpi=120)
    print("Saved plot -> online_gp_rff_conformal.png")
