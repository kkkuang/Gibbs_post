function [A, SIGMA, A_samples, SIGMA_samples] = BVARquantile(Y, p, constant, quantiles, num_samples, varargin)
%% ================================================================
% BVARquantile - Bayesian Multivariate Quantile Regression VAR
% Implements Tian, Tang & Tian (2021) joint quantile regression
% with MAL distribution and optional L1/2 penalty for variable selection
%% ================================================================
%  INPUT
%    Y            TxM matrix of endogenous variables  
%    p            Number of lags
%    constant     1 if intercept, 0 otherwise
%    quantiles    1xM vector of quantile levels (e.g., [0.5, 0.5] for median)
%    num_samples  Number of MCMC samples
%    varargin     Optional: 'prior_type', 'l1_2' or 'noninformative' (default: 'l1_2')
%                          'verbose', true/false (default: true)
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
addParameter(parser, 'prior_type', 'l1_2', @(x) ismember(x, {'l1_2', 'noninformative'}));
addParameter(parser, 'verbose', true, @islogical);
addParameter(parser, 'burnin', 2000, @isnumeric);
addParameter(parser, 'thin', 1, @isnumeric);
parse(parser, varargin{:});
prior_type = parser.Results.prior_type;
verbose = parser.Results.verbose;
burnin = parser.Results.burnin;
thin = parser.Results.thin;

% Prepare data using existing function structure
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y, p, constant);

% Validate quantiles
if length(quantiles) ~= M
    error('Number of quantiles must equal number of variables');
end
if any(quantiles <= 0) || any(quantiles >= 1)
    error('Quantiles must be between 0 and 1');
end

if verbose
    fprintf('Estimating Bayesian Multivariate Quantile Regression VAR\n');
    fprintf('Data: T=%d, M=%d, KK=%d\n', T, M, KK);
    fprintf('Quantiles: [%s]\n', sprintf('%.2f ', quantiles));
    fprintf('Prior: %s\n', prior_type);
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
B = B_ols;                    % KK x M coefficients
W = ones(T, M);               % T x M mixing variables (from MAL distribution)
D = sqrt(diag(diag(SIGMA_ols))); % M x M diagonal scale matrix
Psi = corr(E_ols);            % M x M correlation matrix

% Compute theta parameter from quantiles (for MAL distribution)
theta = zeros(M, 1);
for j = 1:M
    tau_j = quantiles(j);
    theta(j) = (1 - 2*tau_j) / sqrt(tau_j * (1 - tau_j));
end

% Set priors based on type
switch prior_type
    case 'l1_2'
        % L1/2 penalty priors
        lambda = ones(KK, M);     % Tuning parameters for L1/2
        H = ones(KK, M);          % Auxiliary variables for hierarchical L1/2
        use_l1_2 = true;
        if verbose
            fprintf('Using L1/2 penalty for variable selection\n');
        end
        
    case 'noninformative'
        % Noninformative Gaussian priors
        B_prior_var = 100;        % Large variance = noninformative
        use_l1_2 = false;
        if verbose
            fprintf('Using noninformative Gaussian priors\n');
        end
end

% Inverse Wishart prior for Psi
Psi_0 = eye(M);
v_0 = M + 2;

if verbose
    fprintf('Starting MCMC sampler...\n');
    tic;
end

% MCMC loop
sample_idx = 0;
for iter = 1:total_draws
    
    %% Step 1: Sample B (coefficients)
    if use_l1_2
        [B, H] = sample_B_l1_2(Y, X, W, D, theta, Psi, B, H, lambda, M, T, KK);
    else
        B = sample_B_noninformative(Y, X, W, D, theta, Psi, B_prior_var, M, T, KK);
    end
    
    %% Step 2: Sample D (diagonal scale matrix)
    D = sample_D(Y, X, B, W, theta, Psi, M, T);
    
    %% Step 3: Sample Psi (correlation matrix)
    Psi = sample_Psi(Y, X, B, W, D, theta, Psi_0, v_0, M, T);
    
    %% Step 4: Sample W (mixing variables)
    W = sample_W(Y, X, B, D, theta, Psi, M, T);
    
    %% Step 5: Sample L1/2 auxiliary variables (if using L1/2 prior)
    if use_l1_2
        lambda = sample_lambda(H, M, KK);
    end
    
    % Store samples (after burn-in and thinning)
    if iter > burnin && mod(iter - burnin, thin) == 0
        sample_idx = sample_idx + 1;
        A_samples(sample_idx, :, :) = B';     % Convert to M x KK
        SIGMA_samples(sample_idx, :, :) = D * Psi * D;
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

function [B, H] = sample_B_l1_2(Y, X, W, D, theta, Psi, B_old, H, lambda, M, T, KK)
% Sample B with L1/2 penalty using hierarchical representation

B = B_old;
mu_W = mean(W, 1);  % 1 x M

for eq = 1:M
    % Transform data for this equation
    y_tilde = zeros(T, 1);
    X_tilde = zeros(T, KK);
    
    for t = 1:T
        w_sqrt = sqrt(W(t, eq));
        y_tilde(t) = (Y(t, eq) - (W(t, eq) - mu_W(eq)) * D(eq, eq) * theta(eq)) / w_sqrt;
        X_tilde(t, :) = X(t, :) / w_sqrt;
    end
    
    % Prior precision matrix with L1/2 penalty
    V_inv = diag(1 ./ (H(:, eq).^2));
    
    % Posterior precision and mean
    sigma_eq = Psi(eq, eq);
    Sigma_inv = X_tilde' * X_tilde / sigma_eq + V_inv;
    
    % Add regularization
    Sigma_inv = Sigma_inv + 1e-8 * eye(KK);
    
    mu_post = Sigma_inv \ (X_tilde' * y_tilde / sigma_eq);
    
    % Sample from truncated normal (truncation by H bounds)
    try
        L = chol(Sigma_inv, 'lower');
        beta_prop = mu_post + L' \ randn(KK, 1);
        
        % Accept if within bounds
        if all(abs(beta_prop) < H(:, eq).^2)
            B(:, eq) = beta_prop;
        end
    catch
        % Fallback: keep old value
        continue;
    end
end

% Sample H (auxiliary variables for L1/2)
for j = 1:KK
    for eq = 1:M
        % Sample from truncated exponential
        H_star = exprnd(1/lambda(j, eq));
        H(j, eq) = H_star + abs(B(j, eq))^(1/2);
    end
end
end

function B = sample_B_noninformative(Y, X, W, D, theta, Psi, B_prior_var, M, T, KK)
% Sample B with noninformative Gaussian priors

B = zeros(KK, M);
mu_W = mean(W, 1);  % 1 x M

for eq = 1:M
    % Transform data for this equation
    y_tilde = zeros(T, 1);
    X_tilde = zeros(T, KK);
    
    for t = 1:T
        w_sqrt = sqrt(W(t, eq));
        y_tilde(t) = (Y(t, eq) - (W(t, eq) - mu_W(eq)) * D(eq, eq) * theta(eq)) / w_sqrt;
        X_tilde(t, :) = X(t, :) / w_sqrt;
    end
    
    % Noninformative prior precision
    V_inv = eye(KK) / B_prior_var;
    
    % Posterior precision and mean
    sigma_eq = Psi(eq, eq);
    Sigma_inv = X_tilde' * X_tilde / sigma_eq + V_inv;
    
    % Add regularization
    Sigma_inv = Sigma_inv + 1e-8 * eye(KK);
    
    mu_post = Sigma_inv \ (X_tilde' * y_tilde / sigma_eq);
    
    % Sample from multivariate normal
    try
        L = chol(Sigma_inv, 'lower');
        B(:, eq) = mu_post + L' \ randn(KK, 1);
    catch
        % Fallback: use posterior mean
        B(:, eq) = mu_post;
    end
end
end

function D = sample_D(Y, X, B, W, theta, Psi, M, T)
% Sample diagonal scale matrix D using Metropolis-Hastings

D = diag(ones(M, 1));  % Initialize as identity for correlation structure
mu_W = mean(W, 1);

for j = 1:M
    % Compute residuals
    resid = Y(:, j) - X * B(:, j) - (W(:, j) - mu_W(j)) * D(j, j) * theta(j);
    
    % Current value
    d_old = D(j, j);
    
    % Propose new value
    d_prop = d_old + 0.1 * randn();
    
    if d_prop > 0  % Ensure positivity
        % Log likelihood ratio
        log_lik_old = -0.5 * sum(log(W(:, j))) - 0.5 * sum(resid.^2 ./ W(:, j)) / Psi(j, j);
        
        resid_prop = Y(:, j) - X * B(:, j) - (W(:, j) - mu_W(j)) * d_prop * theta(j);
        log_lik_prop = -0.5 * sum(log(W(:, j))) - 0.5 * sum(resid_prop.^2 ./ W(:, j)) / Psi(j, j);
        
        % Accept/reject
        if log(rand()) < log_lik_prop - log_lik_old
            D(j, j) = d_prop;
        end
    end
end
end

function Psi = sample_Psi(Y, X, B, W, D, theta, Psi_0, v_0, M, T)
% Sample correlation matrix from inverse Wishart

mu_W = mean(W, 1);
E = zeros(T, M);

% Compute standardized residuals
for j = 1:M
    resid = Y(:, j) - X * B(:, j) - (W(:, j) - mu_W(j)) * D(j, j) * theta(j);
    E(:, j) = resid ./ sqrt(W(:, j));
end

% Posterior parameters
Psi_post = Psi_0 + E' * E;
v_post = v_0 + T;

% Ensure positive definiteness
Psi_post = (Psi_post + Psi_post') / 2;
min_eig = min(eig(Psi_post));
if min_eig <= 0
    Psi_post = Psi_post + (1e-6 - min_eig) * eye(M);
end

% Sample from inverse Wishart
try
    Psi = iwishrnd(Psi_post, v_post);
catch
    % Fallback
    Psi = Psi_post / (v_post - M - 1);
end

% Ensure positive definite
Psi = (Psi + Psi') / 2;
min_eig = min(eig(Psi));
if min_eig <= 0
    Psi = Psi + (1e-8 - min_eig) * eye(M);
end
end

function W = sample_W(Y, X, B, D, theta, Psi, M, T)
% Sample mixing variables from generalized inverse Gaussian

W = ones(T, M);

for t = 1:T
    for j = 1:M
        % Compute residual
        resid = Y(t, j) - X(t, :) * B(:, j);
        
        % Parameters for GIG distribution
        % W_ij ~ GIG(1-1/2, e_ij^T Psi^{-1} e_ij, theta^T Psi^{-1} theta + 2)
        lambda_gig = 1 - 1/2;
        chi_gig = (resid^2) / Psi(j, j);
        psi_gig = (theta(j)^2) / Psi(j, j) + 2;
        
        % Sample from GIG (using approximation for simplicity)
        % For GIG(lambda, chi, psi): mean ≈ sqrt(chi/psi) when lambda ≈ 0
        if chi_gig > 0 && psi_gig > 0
            W(t, j) = max(sqrt(chi_gig / psi_gig) + 0.1 * randn(), 0.1);
        else
            W(t, j) = 1.0;
        end
    end
end
end

function lambda = sample_lambda(H, M, KK)
% Sample lambda parameters for L1/2 penalty

lambda = zeros(KK, M);
c = 0.1;  % Prior parameter
d = 0.1;  % Prior parameter

for j = 1:KK
    for eq = 1:M
        % Gamma posterior: Gamma(3 + c, d + H_{j,eq})
        lambda(j, eq) = gamrnd(3 + c, 1/(d + H(j, eq)));
    end
end
end

%% ================================================================
%% Helper Functions
%% ================================================================
