# %%
from matplotlib.image import imread
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Online learning: update the data point by point or by mini-batch
def online_update(X_train, y_train, X_new, y_new):
    X_train = np.vstack([X_train, X_new])
    y_train = np.append(y_train, y_new)
    return X_train, y_train

# %%
# Generate simulated dataset
np.random.seed(42)
X_all = np.random.rand(10000, 1) * 10  # Generate 1000 one-dimensional input data points
y_all = np.sin(X_all[:, 0]) + np.random.randn(10000) * 0.1  # Noisy target values
# Initial data: use the first 5 points as the initial training set
X_train = X_all[:50]
y_train = y_all[:50]

# %%
def kernel_rff_approx(X, V, D):
    VX=np.dot(X,V)
    Z = np.sqrt( 1/ D) * np.concatenate([np.sin(VX), np.cos(VX)],axis=1)
    return Z
def gp_predict_rff(Z_train, y_train, Z_test, sigma_n):
    K = Z_train.T @ Z_train +sigma_n**2 * np.eye(len(Z_train.T))/sigma_f**2
    K_inv = np.linalg.inv(K)
    mu_s = Z_test @ K_inv @ Z_train.T @ y_train
    cov_s = Z_test @(sigma_n**2*K_inv) @ Z_test.T + sigma_n**2 *np.eye(len(Z_test))
    return mu_s, cov_s

# %%
def log_likelihood(X, mu_list, sigma_list):
    N = len(X)  

    
    log_likelihoods = []

    for i in range(N):
        
        mu = mu_list[i]
        sigma = sigma_list[i]

        
        log_likelihood_i = -(0.5 * (np.log(2 * np.pi * sigma))+ ((X[i] - mu) ** 2)/ (2*sigma))

        
        log_likelihoods.append(log_likelihood_i)

    return log_likelihoods

# %% [markdown]
# RBF kennel

# %%
length_scale=1
sigma_f = 1
sigma_n = 0.1
D = 300
n_features = X_train.shape[1]
V = np.random.normal(0, np.sqrt(1 / length_scale**2), size=(n_features, D))
# Compute the initial random Fourier features for the training data
Z_train= kernel_rff_approx(X_train,V, D)

# %%
predictions_rbf = []
true_values_rbf = []
variance_rbf=[]
batch_size = 50

# %%
# Start the online learning process, updating one point at a time
for i in tqdm(range(50, len(X_all), batch_size)):
    X_new = X_all[i:i + batch_size]
    y_new = y_all[i:i + batch_size]

    # Transform the new data points using RFF
    Z_new = kernel_rff_approx(X_new,V, D)

    # Perform prediction using RFF (for each new point)
    mu_s, cov_s = gp_predict_rff(Z_train, y_train, Z_new, sigma_n)

    # Store the predictions and the true values
    predictions_rbf.extend(mu_s)
    true_values_rbf.extend(y_new)
    variance_rbf.extend(np.diag(cov_s))

    # Add the true values to the training set
    X_train, y_train = online_update(X_train, y_train, X_new, y_new)

    # Update the RFF-transformed training data
    Z_train = np.vstack([Z_train, Z_new])

# %%
neg_log_likelihood_values_rbf = log_likelihood(true_values_rbf, predictions_rbf, variance_rbf)
likelihoods_rbf = np.exp(-np.array(neg_log_likelihood_values_rbf))

# %% [markdown]
# Laplacian Kernel

# %%
from scipy.stats import cauchy

# %%
length_scale=1
sigma_f = 1.5
sigma_n = 0.1
D = 300
n_features = X_train.shape[1]
V =cauchy.rvs(loc=0, scale=length_scale, size=(n_features, D))
# Compute the initial random Fourier features for the training data
Z_train= kernel_rff_approx(X_train,V, D)

# %%
predictions_lap = []
true_values_lap = []
variance_lap=[]

# %%
# Start the online learning process, updating one point at a time
for i in tqdm(range(50, len(X_all), batch_size)):
    X_new = X_all[i:i + batch_size]
    y_new = y_all[i:i + batch_size]

    # Transform the new data points using RFF
    Z_new = kernel_rff_approx(X_new,V, D)

    # Perform prediction using RFF (for each new point)
    mu_s, cov_s = gp_predict_rff(Z_train, y_train, Z_new, sigma_n)

    # Store the predictions and the true values
    predictions_lap.extend(mu_s)
    true_values_lap.extend(y_new)
    variance_lap.extend(np.diag(cov_s))

    # Add the true values to the training set
    X_train, y_train = online_update(X_train, y_train, X_new, y_new)

    # Update the RFF-transformed training data
    Z_train = np.vstack([Z_train, Z_new])

# %%


# %%
neg_log_likelihood_values_lap = log_likelihood(true_values_lap, predictions_lap, variance_lap)
likelihoods_lap = np.exp(-np.array(neg_log_likelihood_values_lap))

# %% [markdown]
# Matern Kernel

# %%
from scipy.stats import gamma, multivariate_normal

# %%
length_scale=1
sigma_f = 1.5
sigma_n = 0.1
D = 300
n_features = X_train.shape[1]
nu_matern = 1.5    


nu = 2 * nu_matern  
mean = np.zeros(D)
scale = (2 * nu) / (length_scale ** 2)
scale_matrix = scale * np.eye(D)

g = gamma.rvs(a=nu/2, scale=2/nu, size=n_features)

z = multivariate_normal.rvs(mean, scale_matrix,n_features)

V= z / np.sqrt(g)[:, None]

# Compute the initial random Fourier features for the training data
Z_train = kernel_rff_approx(X_train,V, D)

# %%
predictions_matern = []
true_values_matern = []
variance_matern=[]
batch_size = 50

# %%
# Start the online learning process, updating one point at a time
for i in tqdm(range(50, len(X_all), batch_size)):
    X_new = X_all[i:i + batch_size]
    y_new = y_all[i:i + batch_size]

    # Transform the new data points using RFF
    Z_new = kernel_rff_approx(X_new,V, D)

    # Perform prediction using RFF (for each new point)
    mu_s, cov_s = gp_predict_rff(Z_train, y_train, Z_new, sigma_n)

    # Store the predictions and the true values
    predictions_matern.extend(mu_s)
    true_values_matern.extend(y_new)
    variance_matern.extend(np.diag(cov_s))

    # Add the true values to the training set
    X_train, y_train = online_update(X_train, y_train, X_new, y_new)

    # Update the RFF-transformed training data
    Z_train = np.vstack([Z_train, Z_new])

# %%
neg_log_likelihood_values_matern = log_likelihood(true_values_matern, predictions_matern, variance_matern)
likelihoods_matern = np.exp(-np.array(neg_log_likelihood_values_matern))

# %% [markdown]
# Rational Quadratic Kernel

# %%
from scipy.stats import gamma, multivariate_normal

def sample_rq_spectral(alpha, length_scale, D, n_features=1000):
    
    nu = 2 * alpha                            
    mean = np.zeros(D)                        
    scale = (length_scale**2)/(2*alpha)       
    scale_matrix = scale * np.eye(D)          
    
  
    g = gamma.rvs(a=alpha, scale=1/alpha, size=n_features)

    z = multivariate_normal.rvs(mean, scale_matrix, n_features)
    
    
    V = z / np.sqrt(g)[:, None]
    
    return V


alpha = 2.0          
length_scale = 1.0   
sigma_f = 1.5
sigma_n = 0.1
D = 300
n_features = X_train.shape[1]
V = sample_rq_spectral(alpha, length_scale, D, n_features)
Z_train = kernel_rff_approx(X_train,V, D)

# %%
predictions_quad = []
true_values_quad = []
variance_quad=[]
batch_size = 50

# %%
# Start the online learning process, updating one point at a time
for i in tqdm(range(50, len(X_all), batch_size)):
    X_new = X_all[i:i + batch_size]
    y_new = y_all[i:i + batch_size]

    # Transform the new data points using RFF
    Z_new = kernel_rff_approx(X_new,V, D)

    # Perform prediction using RFF (for each new point)
    mu_s, cov_s = gp_predict_rff(Z_train, y_train, Z_new, sigma_n)

    # Store the predictions and the true values
    predictions_quad.extend(mu_s)
    true_values_quad.extend(y_new)
    variance_quad.extend(np.diag(cov_s))

    # Add the true values to the training set
    X_train, y_train = online_update(X_train, y_train, X_new, y_new)

    # Update the RFF-transformed training data
    Z_train = np.vstack([Z_train, Z_new])

# %%
neg_log_likelihood_values_quad = log_likelihood(true_values_quad, predictions_quad, variance_quad)
likelihoods_quad = np.exp(-np.array(neg_log_likelihood_values_quad))

# %% [markdown]
# Ensemble Guassian Process

# %%

neg_log_likelihoods_quad = np.array(neg_log_likelihood_values_quad)
neg_log_likelihoods_matern = np.array(neg_log_likelihood_values_matern)
neg_log_likelihoods_lap = np.array(neg_log_likelihood_values_lap)
neg_log_likelihoods_rbf = np.array(neg_log_likelihood_values_rbf)


weights = np.array([1/4, 1/4, 1/4, 1/4])

all_weights = []  
for t in range(len(neg_log_likelihoods_rbf)):
    
    l_quad = neg_log_likelihoods_quad[t]
    l_matern = neg_log_likelihoods_matern[t]
    l_lap = neg_log_likelihoods_lap[t]
    l_rbf = neg_log_likelihoods_rbf[t]

    
    exp_terms = np.array([
        np.exp(l_quad - l_rbf),    
        np.exp(l_matern - l_rbf),  
        np.exp(l_lap - l_rbf),    
        1                          
    ])

    
    current_weights = weights * exp_terms
    current_weights /= current_weights.sum()

    
    all_weights.append(current_weights)

    
    weights = current_weights

all_weights = np.array(all_weights)

print("All time weights:", all_weights)

# %%
import numpy as np

# Negative log-likelihood values converted to NumPy arrays
neg_log_likelihoods_quad = np.array(neg_log_likelihood_values_quad)
neg_log_likelihoods_matern = np.array(neg_log_likelihood_values_matern)
neg_log_likelihoods_lap = np.array(neg_log_likelihood_values_lap)
neg_log_likelihoods_rbf = np.array(neg_log_likelihood_values_rbf)

# Convert negative log-likelihood values to probability values (without normalization)
probabilities_quad = np.exp(-neg_log_likelihoods_quad)
probabilities_matern = np.exp(-neg_log_likelihoods_matern)
probabilities_lap = np.exp(-neg_log_likelihoods_lap)
probabilities_rbf = np.exp(-neg_log_likelihoods_rbf)

# Matrix of probability predictions for each kernel at all time points
probabilities_matrix = np.vstack([probabilities_quad, probabilities_matern, probabilities_lap, probabilities_rbf]).T

# Apply weights
# all_weights should be an N x 4 matrix, where N is the number of time points and 4 is the number of kernels
weighted_probs = np.sum(all_weights * probabilities_matrix, axis=1)

print("Weighted prediction probabilities:", weighted_probs)

# %%
predictions_set=np.stack([predictions_quad, predictions_matern, predictions_lap, predictions_rbf]).T
ensemble_predictions = np.sum(all_weights * predictions_set, axis=1)
print("Ensemble predictions:", ensemble_predictions)

# %%
# Plot all predictions vs true values at the end
# Plot all predictions vs true values at the end
plt.figure(figsize=(12, 6))
plt.plot(range(100,300), true_values_lap[100:300], 'ro', label='True Values')
plt.plot(range(100,300), ensemble_predictions[100:300], 'bx-', label='Predictions')
plt.title("Gaussian Process Regression with Online Learning (Optimized Laplacian Approximation)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()

# %%
from sklearn.metrics import mean_squared_error
err_quad = mean_squared_error(true_values_quad, predictions_quad)
err_matern = mean_squared_error(true_values_matern, predictions_matern)
err_lap = mean_squared_error(true_values_lap, predictions_lap)
err_rbf = mean_squared_error(true_values_rbf, predictions_rbf)
err_ensemble = mean_squared_error(true_values_lap, ensemble_predictions)
print("MSE for Quadratic Approximation:", err_quad)
print("MSE for Matern Approximation:", err_matern)
print("MSE for Laplacian Approximation:", err_lap)
print("MSE for RBF Approximation:", err_rbf)
print("MSE for Ensemble Model:", err_ensemble)


