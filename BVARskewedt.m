function [A, SIGMA, A_samples, SIGMA_samples] = BVARskewedt(Y, p, constant, num_samples, varargin)
%% ================================================================
% BVARskewedt - Bayesian VAR with Skewed-t Innovations  
% Implements multivariate skewed-t errors for outlier robustness
% Based on Karlsson, Mazur & Nguyen (JEDC 2023) - no stochastic volatility
%% ================================================================
%  INPUT
%    Y            TxM matrix of endogenous variables  
%    p            Number of lags
%    constant     1 if intercept, 0 otherwise
%    num_samples  Number of MCMC samples
%    varargin     Optional: 'verbose', true/false (default: true)
%                          'burnin', scalar (default: 2000)
%                          'thin', scalar (default: 1)
%% ================================================================
%  OUTPUT  
%    A            Posterior mean of VAR coefficients (KK x M)
%    SIGMA        Posterior mean of covariance matrix (M x M)
%    A_samples    All posterior samples of A (num_samples x M x KK)
%    SIGMA_samples All posterior samples of SIGMA (num_samples x M x M)
%% ================================================================

% Parse inputs
parser = inputParser;
addParameter(parser, 'verbose', true, @islogical);
addParameter(parser, 'burnin', 2000, @isnumeric);
addParameter(parser, 'thin', 1, @isnumeric);
parse(parser, varargin{:});
verbose = parser.Results.verbose;
burnin = parser.Results.burnin;
thin = parser.Results.thin;

% Prepare data using existing function structure
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y, p, constant);

if verbose
    fprintf('Estimating Bayesian VAR with Skewed-t Errors\n');
    fprintf('Data: T=%d, M=%d, KK=%d\n', T, M, KK);
    fprintf('MCMC: %d samples + %d burn-in (thinning: %d)\n', num_samples, burnin, thin);
end

% Initialize storage
total_draws = burnin + num_samples * thin;
A_samples = zeros(num_samples, M, KK);
SIGMA_samples = zeros(num_samples, M, M);

% Starting values - OLS estimates
B_ols = (X' * X) \ (X' * Y);  % KK x M
E_ols = Y - X * B_ols;
SIGMA_ols = (E_ols' * E_ols) / T;

% Initialize parameters
B = B_ols;              % KK x M coefficients
gamma = zeros(M, 1);    % skewness parameters
nu = 10 * ones(M, 1);   % degrees of freedom (>4 for finite variance)
SIGMA = SIGMA_ols;      % residual covariance
xi = ones(T, M);        % mixing variables

% Priors (noninformative)
% B: diffuse normal N(0, 100*I)
B_prior_mean = zeros(KK, M);
B_prior_var = 100;

% gamma: normal N(0, 10)
gamma_prior_var = 10;

% nu: truncated gamma G(2, 0.1) on [4.1, 100]
nu_prior_a = 2;
nu_prior_b = 0.1;
nu_min = 4.1;
nu_max = 100;

% SIGMA: inverse Wishart IW(M+2, I)
Psi_0 = eye(M);
v_0 = M + 2;

if verbose
    fprintf('Starting MCMC sampler...\n');
    tic;
end

% MCMC loop
sample_idx = 0;
for iter = 1:total_draws
    
    %% Step 1: Sample B and gamma jointly
    [B, gamma] = sample_B_gamma(Y, X, xi, B, gamma, M, T, KK, B_prior_mean, B_prior_var, gamma_prior_var, SIGMA);
    
    %% Step 2: Sample SIGMA 
    SIGMA = sample_SIGMA(Y, X, B, gamma, xi, Psi_0, v_0, M, T);
    
    %% Step 3: Sample nu (degrees of freedom)
    nu = sample_nu(xi, nu, nu_prior_a, nu_prior_b, nu_min, nu_max, M, T);
    
    %% Step 4: Sample xi (mixing variables)
    xi = sample_xi(Y, X, B, gamma, SIGMA, nu, xi, M, T);
    
    % Store samples (after burn-in and thinning)
    if iter > burnin && mod(iter - burnin, thin) == 0
        sample_idx = sample_idx + 1;
        A_samples(sample_idx, :, :) = B';     % Convert to M x KK
        SIGMA_samples(sample_idx, :, :) = SIGMA;
    end
    
    if verbose && mod(iter, 1000) == 0
        fprintf('Iteration %d/%d\n', iter, total_draws);
    end
end

if verbose
    fprintf('MCMC completed in %.2f seconds\n', toc);
end

% Posterior means
A = squeeze(mean(A_samples, 1))';      % KK x M
SIGMA = squeeze(mean(SIGMA_samples, 1)); % M x M

end

%% ================================================================
%% MCMC Sampling Functions
%% ================================================================

function [B, gamma] = sample_B_gamma(Y, X, xi, B, gamma, M, T, KK, B_prior_mean, B_prior_var, gamma_prior_var, SIGMA)
% Sample B and gamma jointly maintaining multivariate structure

mu_xi = mean(xi, 1);  % 1 x M

% Transform data using mixing variables - more stable approach
Y_trans = zeros(T, M);
X_B_trans = zeros(T, KK, M);  % T x KK x M
X_gamma_trans = zeros(T, M);   % T x M

for t = 1:T
    for i = 1:M
        xi_sqrt = sqrt(xi(t, i));
        Y_trans(t, i) = Y(t, i) / xi_sqrt;
        X_B_trans(t, :, i) = X(t, :) / xi_sqrt;
        X_gamma_trans(t, i) = (xi(t, i) - mu_xi(i)) / xi_sqrt;
    end
end

% Stack into system form more carefully
Y_vec = Y_trans(:);  % TM x 1
X_system = zeros(T*M, KK*M + M);

% Fill design matrix
for t = 1:T
    for i = 1:M
        row_idx = (t-1)*M + i;
        
        % B coefficients (each equation gets its own KK coefficients)
        col_start_B = (i-1)*KK + 1;
        col_end_B = i*KK;
        X_system(row_idx, col_start_B:col_end_B) = X_B_trans(t, :, i);
        
        % gamma coefficients (each equation gets its own gamma)
        col_gamma = KK*M + i;
        X_system(row_idx, col_gamma) = X_gamma_trans(t, i);
    end
end

% Prior precision matrix (more stable construction)
total_params = KK*M + M;
V_prior_inv = zeros(total_params, total_params);

% B coefficients - each equation separate
for i = 1:M
    idx_start = (i-1)*KK + 1;
    idx_end = i*KK;
    V_prior_inv(idx_start:idx_end, idx_start:idx_end) = eye(KK) / B_prior_var;
end

% gamma coefficients
for i = 1:M
    idx_gamma = KK*M + i;
    V_prior_inv(idx_gamma, idx_gamma) = 1 / gamma_prior_var;
end

% Compute posterior precision more stably
SIGMA_inv = inv(SIGMA);
Omega = kron(SIGMA_inv, eye(T));

% Use more stable computation
XtOmega = X_system' * Omega;
V_post_inv = XtOmega * X_system + V_prior_inv;

% Add regularization to ensure positive definiteness
V_post_inv = V_post_inv + 1e-10 * eye(total_params);

% Posterior mean
b_post = XtOmega * Y_vec;
mu_post = V_post_inv \ b_post;

% Sample using Cholesky with better error handling
try
    % Try standard Cholesky
    L = chol(V_post_inv, 'lower');
    theta_sample = mu_post + L' \ randn(total_params, 1);
catch
    % If Cholesky fails, use eigenvalue regularization
    [V, D] = eig(V_post_inv);
    d = real(diag(D));
    d = max(d, 1e-8);  % Regularize eigenvalues
    V_post_inv_reg = V * diag(d) * V';
    V_post_inv_reg = (V_post_inv_reg + V_post_inv_reg') / 2;  % Ensure symmetry
    
    L = chol(V_post_inv_reg, 'lower');
    theta_sample = mu_post + L' \ randn(total_params, 1);
end

% Extract B and gamma (maintaining system structure)
B = zeros(KK, M);
gamma = zeros(M, 1);

for i = 1:M
    % Extract B coefficients for equation i
    idx_start = (i-1)*KK + 1;
    idx_end = i*KK;
    B(:, i) = theta_sample(idx_start:idx_end);
    
    % Extract gamma for equation i
    idx_gamma = KK*M + i;
    gamma(i) = theta_sample(idx_gamma);
end
end

function SIGMA = sample_SIGMA(Y, X, B, gamma, xi, Psi_0, v_0, M, T)
% Sample SIGMA from inverse Wishart posterior

% Compute residuals
E = Y - X * B;

% Adjust for skewness and mixing variables
mu_xi = mean(xi, 1);
E_adjusted = zeros(T, M);

for t = 1:T
    for i = 1:M
        % Adjust for skewness term
        skew_adjust = (xi(t, i) - mu_xi(i)) * gamma(i);
        % Scale by sqrt(xi) to account for variance scaling
        E_adjusted(t, i) = (E(t, i) - skew_adjust) / sqrt(xi(t, i));
    end
end

% Posterior parameters
Psi_post = Psi_0 + E_adjusted' * E_adjusted;
v_post = v_0 + T;

% Ensure Psi_post is positive definite
Psi_post = (Psi_post + Psi_post') / 2;  % Ensure symmetry
min_eig = min(eig(Psi_post));
if min_eig <= 0
    Psi_post = Psi_post + (1e-6 - min_eig) * eye(M);
end

% Sample from inverse Wishart
try
    SIGMA = iwishrnd(Psi_post, v_post);
catch
    % Fallback: use classical estimate with small regularization
    SIGMA = E_adjusted' * E_adjusted / T + 1e-6 * eye(M);
end

% Ensure SIGMA is positive definite
SIGMA = (SIGMA + SIGMA') / 2;
min_eig = min(eig(SIGMA));
if min_eig <= 0
    SIGMA = SIGMA + (1e-6 - min_eig) * eye(M);
end
end

function nu = sample_nu(xi, nu_old, a_prior, b_prior, nu_min, nu_max, M, T)
% Sample degrees of freedom using Metropolis-Hastings

nu = nu_old;

for i = 1:M
    % Propose new value
    nu_prop = nu(i) + 0.5 * randn();
    
    % Check bounds
    if nu_prop < nu_min || nu_prop > nu_max
        continue;
    end
    
    % Log likelihood ratio
    log_lik_old = T * (gammaln(nu(i)/2) - (nu(i)/2) * log(nu(i)/2)) - ...
                  (nu(i)/2 + 1) * sum(log(xi(:, i))) - (nu(i)/2) * sum(1./xi(:, i));
    
    log_lik_prop = T * (gammaln(nu_prop/2) - (nu_prop/2) * log(nu_prop/2)) - ...
                   (nu_prop/2 + 1) * sum(log(xi(:, i))) - (nu_prop/2) * sum(1./xi(:, i));
    
    % Log prior ratio
    log_prior_old = (a_prior - 1) * log(nu(i)) - b_prior * nu(i);
    log_prior_prop = (a_prior - 1) * log(nu_prop) - b_prior * nu_prop;
    
    % Accept/reject
    log_alpha = log_lik_prop - log_lik_old + log_prior_prop - log_prior_old;
    
    if log(rand()) < log_alpha
        nu(i) = nu_prop;
    end
end
end

function xi_new = sample_xi(Y, X, B, gamma, SIGMA, nu, xi_old, M, T)
% Sample mixing variables using Metropolis-Hastings

xi_new = xi_old;

for t = 1:T
    for i = 1:M
        % Current mu_xi for this variable (needed for skewness adjustment)
        mu_xi_i = mean(xi_old(:, i));
        
        % Compute residual for current xi
        resid_old = Y(t, i) - X(t, :) * B(:, i) - (xi_old(t, i) - mu_xi_i) * gamma(i);
        
        % Propose new xi value using random walk on log scale
        log_xi_old = log(xi_old(t, i));
        log_xi_prop = log_xi_old + 0.1 * randn();  % Random walk proposal
        xi_prop = exp(log_xi_prop);
        
        % Ensure xi_prop is positive and reasonable
        if xi_prop <= 0 || xi_prop > 100
            continue;
        end
        
        % Compute residual for proposed xi
        resid_prop = Y(t, i) - X(t, :) * B(:, i) - (xi_prop - mu_xi_i) * gamma(i);
        
        % Log likelihood ratio (normal part)
        log_lik_old = -0.5 * log(xi_old(t, i)) - 0.5 * (resid_old^2) / (xi_old(t, i) * SIGMA(i, i));
        log_lik_prop = -0.5 * log(xi_prop) - 0.5 * (resid_prop^2) / (xi_prop * SIGMA(i, i));
        
        % Log prior ratio (IG(nu/2, nu/2))
        log_prior_old = -(nu(i)/2 + 1) * log(xi_old(t, i)) - (nu(i)/2) / xi_old(t, i);
        log_prior_prop = -(nu(i)/2 + 1) * log(xi_prop) - (nu(i)/2) / xi_prop;
        
        % Jacobian for log-scale proposal
        log_jacobian = log_xi_prop - log_xi_old;
        
        % Accept/reject
        log_alpha = log_lik_prop - log_lik_old + log_prior_prop - log_prior_old + log_jacobian;
        
        if log(rand()) < log_alpha
            xi_new(t, i) = xi_prop;
        end
    end
end
end
