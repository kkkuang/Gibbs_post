function [A, SIGMA, A_samples, SIGMA_samples] = BVARlossLLB_joint(Y, p, constant, loss_fun, num_samples, varargin)

%% ================================================================
% BVARlossLLB_joint - TRUE Joint Bayesian VAR with Loss-Likelihood Bootstrap
% OPTIMIZED VERSION WITH ANALYTICAL GRADIENTS - Implements simultaneous optimization of coefficients A and covariance SIGMA
% using a SINGLE joint objective function with analytical gradients for faster convergence
%% ================================================================
%  INPUT
%    Y            TxM matrix of endogenous variables  
%    p            Number of lags
%    constant     1 if intercept, 0 otherwise
%    loss_fun     Loss function: 'ols', 'lms', 'lad', 'huber', 'studentt'
%    num_samples  Number of bootstrap samples
%    varargin     Optional: 'calibrate_w', true/false (default: true)
%                           'kappa1', scalar (default: 1000, noninformative)
%                           'kappa2', scalar (default: 1000, noninformative) 
%                           'kappa3', scalar (default: 1000, noninformative)
%                           'ar_lags', scalar (default: 4, for residual variance)
%% ================================================================
%  OUTPUT  
%    A            Posterior mean of VAR coefficients (KK x M)
%    SIGMA        Posterior mean of covariance matrix (M x M)
%    A_samples    All posterior samples of A (num_samples x M x KK)
%    SIGMA_samples All posterior samples of SIGMA (num_samples x M x M)
%% ================================================================

% Parse inputs
parser = inputParser;
addParameter(parser, 'calibrate_w', true, @islogical);
addParameter(parser, 'parallel', true, @islogical);
addParameter(parser, 'kappa1', 1000, @isnumeric);    % Own lag shrinkage
addParameter(parser, 'kappa2', 1000, @isnumeric);    % Other lag shrinkage  
addParameter(parser, 'kappa3', 1000, @isnumeric);    % Intercept shrinkage
addParameter(parser, 'ar_lags', 4, @isnumeric);      % AR lags for residual variance
addParameter(parser, 'verbose', true, @islogical);   % Verbose output
addParameter(parser, 'max_iter', 500, @isnumeric);   % Reduced default max iterations
addParameter(parser, 'tol', 1e-5, @isnumeric);       % Tighter tolerance with analytical gradients
parse(parser, varargin{:});

calibrate_w = parser.Results.calibrate_w;
use_parallel = parser.Results.parallel;
kappa1 = parser.Results.kappa1;
kappa2 = parser.Results.kappa2;
kappa3 = parser.Results.kappa3;
ar_lags = parser.Results.ar_lags;
verbose = parser.Results.verbose;
max_iter = parser.Results.max_iter;
tol = parser.Results.tol;

% Prepare data
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y, p, constant);

% Pre-compute commonly used matrices for speed
XtX = X' * X;
XtY = X' * Y;

% Compute residual variances for Minnesota prior
s_squared = compute_residual_variances(Y, ar_lags);

% Prior settings for SIGMA - inverse Wishart
Psi_0 = diag(s_squared); 
v_0 = M + 2;

% Pre-compute Minnesota precision matrix (unchanged across samples)
V_inv_full = build_joint_minnesota_precision(M, p, constant, kappa1, kappa2, kappa3, s_squared);

% Calibration
if calibrate_w && ~strcmpi(loss_fun, 'lms')
    beta = calibrate_loss_scale_fast(Y, X, loss_fun, XtX, XtY);
    if verbose
        fprintf('Calibrated loss scale beta = %.4f\n', beta);
    end
else
    beta = 1.0;
    if strcmpi(loss_fun, 'lms') && verbose
        fprintf('Using beta = 1.0 for LMS (median is scale-invariant)\n');
    end
end

% Pre-allocate
A_samples = zeros(num_samples, M, KK);
SIGMA_samples = zeros(num_samples, M, M);

% Pre-compute optimization options with analytical gradients
options = optimoptions('fminunc', ...
    'Algorithm', 'quasi-newton', ...
    'Display', 'off', ...
    'MaxIterations', max_iter, ...
    'OptimalityTolerance', tol, ...
    'StepTolerance', 1e-10, ...
    'UseParallel', false, ...
    'SpecifyObjectiveGradient', true); % Enable analytical gradients

if verbose
    fprintf('Running JOINT loss-likelihood bootstrap with %s loss and analytical gradients (κ1=%.1f, κ2=%.1f)...\n', ...
            loss_fun, kappa1, kappa2);
end
tic;

%% Main Bootstrap Loop
if use_parallel
    parfor i = 1:num_samples
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = optimized_joint_bootstrap_sample(...
            Y, X, XtX, XtY, M, T, KK, p, constant, beta, loss_fun, ...
            s_squared, Psi_0, v_0, V_inv_full, options);
    end
else
    for i = 1:num_samples
        if verbose && mod(i, 100) == 0
            fprintf('Sample %d/%d\n', i, num_samples); 
        end
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = optimized_joint_bootstrap_sample(...
            Y, X, XtX, XtY, M, T, KK, p, constant, beta, loss_fun, ...
            s_squared, Psi_0, v_0, V_inv_full, options);
    end
end

if verbose
    fprintf('Completed in %.2f seconds\n', toc);
end

% Posterior means
A = squeeze(mean(A_samples, 1))';      % KK x M
SIGMA = squeeze(mean(SIGMA_samples, 1)); % M x M
end

%% ================================================================
%% OPTIMIZED Joint Bootstrap Sample - Single Objective Optimization with Analytical Gradients
%% ================================================================
function [A_sample, SIGMA_sample] = optimized_joint_bootstrap_sample(Y, X, XtX, XtY, M, T, KK, p, constant, ...
    beta, loss_fun, s_squared, Psi_0, v_0, V_inv_full, options)

% 1. Generate Dirichlet weights (vectorized)
weights = gamrnd(ones(T, 1), 1);
weights = weights * (T / sum(weights)); % Faster normalization

% 2. Compute weighted matrices efficiently
sqrt_w = sqrt(weights);
W_sqrt = spdiags(sqrt_w, 0, T, T);
XtW = X' * W_sqrt;
YtW = Y' * W_sqrt;

% Initial estimates using weighted least squares
A_init = (XtW * XtW') \ (XtW * YtW');  % KK x M
E_init = Y - X * A_init;
E_weighted = E_init .* sqrt_w;
S_init = Psi_0 + E_weighted' * E_weighted;

% Improved initial SIGMA estimate
try
    SIGMA_init = iwishrnd(S_init, v_0 + sum(weights));
catch
    % Fallback if iwishrnd fails
    SIGMA_init = S_init / (v_0 + sum(weights) - M - 1);
end

%% 3. OPTIMIZED JOINT OPTIMIZATION WITH ANALYTICAL GRADIENTS
% Use log-Cholesky parameterization for better numerical stability
L_init = chol(SIGMA_init, 'lower');
L_vec_init = extract_log_cholesky(L_init);

% Combine parameters more efficiently
theta_init = [A_init(:); L_vec_init];

% Pre-compute constants for objective function and gradient
obj_constants = struct();
obj_constants.Y = Y;
obj_constants.X = X;
obj_constants.weights = weights;
obj_constants.sqrt_w = sqrt_w;
obj_constants.beta = beta;
obj_constants.V_inv_full = V_inv_full;
obj_constants.Psi_0 = Psi_0;
obj_constants.v_0 = v_0;
obj_constants.loss_fun = loss_fun;
obj_constants.M = M;
obj_constants.KK = KK;
obj_constants.T = T;

% Build optimized objective function with analytical gradient
objective_fun = @(theta) optimized_joint_objective_with_gradient(theta, obj_constants);

% Optimize with better error handling
try
    [theta_opt, ~] = fminunc(objective_fun, theta_init, options);
    
    % Extract optimized parameters
    A_vec_opt = theta_opt(1:KK*M);
    L_vec_opt = theta_opt(KK*M+1:end);
    
    A_opt = reshape(A_vec_opt, KK, M);
    L_opt = reconstruct_log_cholesky(L_vec_opt, M);
    SIGMA_opt = L_opt * L_opt';
    
catch
    % More robust fallback
    A_opt = A_init;
    SIGMA_opt = SIGMA_init;
end

% Ensure SIGMA is well-conditioned with faster eigenvalue cleanup
min_eig = 1e-8;
if min(eig(SIGMA_opt)) < min_eig
    [V, D] = eig(SIGMA_opt);
    D = diag(max(diag(D), min_eig));
    SIGMA_opt = V * D * V';
end

A_sample = A_opt';        % Convert to M x KK
SIGMA_sample = SIGMA_opt; % M x M
end

%% ================================================================
%% OPTIMIZED Joint Objective Function WITH ANALYTICAL GRADIENT
%% ================================================================
function [f, g] = optimized_joint_objective_with_gradient(theta, constants)

% Extract parameters
A_vec = theta(1:constants.KK*constants.M);
L_vec = theta(constants.KK*constants.M+1:end);
M = constants.M;
KK = constants.KK;

% Reshape A
A = reshape(A_vec, KK, M);

% Reconstruct SIGMA from log-Cholesky factor
try
    L = reconstruct_log_cholesky(L_vec, M);
    SIGMA = L * L';
catch
    f = 1e10;
    if nargout > 1
        g = zeros(size(theta));
    end
    return;
end

% Quick numerical stability check
if rcond(SIGMA) < 1e-12
    f = 1e10;
    if nargout > 1
        g = zeros(size(theta));
    end
    return;
end

%% OBJECTIVE FUNCTION COMPUTATION
E = constants.Y - constants.X * A;  % Residuals T x M

% Compute Cholesky decomposition once
try
    L_chol = chol(SIGMA, 'lower');
    L_inv = L_chol \ eye(M);
    SIGMA_inv_sqrt = L_inv';
    SIGMA_inv = SIGMA_inv_sqrt * SIGMA_inv_sqrt';
catch
    f = 1e10;
    if nargout > 1
        g = zeros(size(theta));
    end
    return;
end

% Transform residuals: E_transformed = E * SIGMA_inv_sqrt'
E_transformed = E * SIGMA_inv_sqrt;

% Compute Mahalanobis distances more efficiently
maha_dist = sqrt(sum(E_transformed.^2, 2));

% Vectorized robust loss computation
[loss_values, loss_derivatives] = compute_robust_loss_and_derivatives(maha_dist, constants.loss_fun);

% Weighted data loss
f_data = constants.beta * (constants.weights' * loss_values);

%% Prior Terms
% 1. Minnesota prior on A coefficients
diff_A = A_vec; % Prior mean is zero
f_prior_A = 0.5 * (diff_A' * constants.V_inv_full * diff_A);

% 2. Inverse Wishart prior on SIGMA (using log-det from Cholesky)
log_det_SIGMA = 2 * sum(log(diag(L_chol)));
trace_term = sum(sum((constants.Psi_0 * SIGMA_inv_sqrt) .* SIGMA_inv_sqrt'));

f_prior_SIGMA = 0.5 * (constants.v_0 + M + 1) * log_det_SIGMA + 0.5 * trace_term;

% Total objective (negative log posterior)
f = f_data + f_prior_A + f_prior_SIGMA;

%% ANALYTICAL GRADIENT COMPUTATION
if nargout > 1
    % Initialize gradient
    g = zeros(size(theta));
    
    %% Gradient w.r.t. A parameters
    % Data loss gradient w.r.t. A
    % For robust losses: sum_t w_t * rho'(||e_t||_Sigma) * (1/||e_t||_Sigma) * e_t^T * SIGMA_inv * dE/dA
    
    % Compute gradient weights for each observation
    grad_weights = constants.weights .* loss_derivatives ./ max(maha_dist, 1e-12);
    
    % Gradient of data loss w.r.t. A
    grad_data_A = zeros(KK, M);
    for t = 1:constants.T
        e_t = E(t, :)'; % M x 1
        grad_data_A = grad_data_A - grad_weights(t) * constants.X(t, :)' * e_t' * SIGMA_inv;
    end
    
    grad_data_A_vec = constants.beta * grad_data_A(:);
    
    % Prior gradient w.r.t. A
    grad_prior_A_vec = constants.V_inv_full * diff_A;
    
    % Total gradient w.r.t. A
    g(1:KK*M) = grad_data_A_vec + grad_prior_A_vec;
    
    %% Gradient w.r.t. L parameters (log-Cholesky)
    % This is more complex - we need chain rule through SIGMA = L*L'
    
    % Gradient of data loss w.r.t. SIGMA
    grad_data_SIGMA = zeros(M, M);
    
    for t = 1:constants.T
        e_t = E(t, :)'; % M x 1
        maha_t = maha_dist(t);
        weight_t = constants.weights(t);
        
        if maha_t > 1e-12
            % Derivative of loss w.r.t. Mahalanobis distance
            d_loss_d_maha = loss_derivatives(t);
            
            % Derivative of Mahalanobis distance w.r.t. SIGMA^{-1}
            % d(||e||_Sigma)/d(SIGMA^{-1}) = 0.5 * (1/||e||_Sigma) * e * e'
            d_maha_d_SIGMA_inv = 0.5 * (e_t * e_t') / maha_t;
            
            % Chain rule: d_loss/d_SIGMA = d_loss/d_maha * d_maha/d_SIGMA_inv * d_SIGMA_inv/d_SIGMA
            % Since d_SIGMA_inv/d_SIGMA = -SIGMA^{-1} * dS * SIGMA^{-1}, we get:
            grad_data_SIGMA = grad_data_SIGMA - weight_t * d_loss_d_maha * SIGMA_inv * d_maha_d_SIGMA_inv * SIGMA_inv;
        end
    end
    
    grad_data_SIGMA = constants.beta * grad_data_SIGMA;
    
    % Gradient of inverse Wishart prior w.r.t. SIGMA
    grad_prior_SIGMA = 0.5 * (constants.v_0 + M + 1) * SIGMA_inv - 0.5 * SIGMA_inv * constants.Psi_0 * SIGMA_inv;
    
    % Total gradient w.r.t. SIGMA
    grad_total_SIGMA = grad_data_SIGMA + grad_prior_SIGMA;
    
    % Convert gradient w.r.t. SIGMA to gradient w.r.t. L (log-Cholesky parameters)
    grad_L_vec = compute_gradient_wrt_log_cholesky(grad_total_SIGMA, L, L_vec);
    
    g(KK*M+1:end) = grad_L_vec;
end
end

%% ================================================================
%% ROBUST LOSS FUNCTIONS WITH DERIVATIVES
%% ================================================================
function [loss_vec, deriv_vec] = compute_robust_loss_and_derivatives(u_vec, loss_fun)
% u_vec: vector of distances (non-negative)
% Returns both loss values and their derivatives w.r.t. u

switch lower(loss_fun)
    case 'ols'
        loss_vec = 0.5 * u_vec.^2;
        deriv_vec = u_vec;
        
    case 'lad'
        loss_vec = u_vec;
        deriv_vec = ones(size(u_vec)); % Derivative is 1 (subgradient at 0)
        
    case 'lms'
        loss_vec = u_vec;  % For LMS, we still use L1 norm in optimization
        deriv_vec = ones(size(u_vec));
        
    case 'huber'
        delta = 1.345;  % Standard Huber parameter
        mask = u_vec <= delta;
        loss_vec = zeros(size(u_vec));
        deriv_vec = zeros(size(u_vec));
        
        loss_vec(mask) = 0.5 * u_vec(mask).^2;
        loss_vec(~mask) = delta * (u_vec(~mask) - 0.5 * delta);
        
        deriv_vec(mask) = u_vec(mask);
        deriv_vec(~mask) = delta;
        
    case 'studentt'
        nu = 4;  % Degrees of freedom
        loss_vec = 0.5 * (nu + 1) * log(1 + u_vec.^2 / nu);
        deriv_vec = (nu + 1) * u_vec ./ (nu + u_vec.^2);
        
    otherwise
        % Default to OLS
        loss_vec = 0.5 * u_vec.^2;
        deriv_vec = u_vec;
end
end

%% ================================================================
%% GRADIENT W.R.T. LOG-CHOLESKY PARAMETERS
%% ================================================================
function grad_L_vec = compute_gradient_wrt_log_cholesky(grad_SIGMA, L, L_vec)
% Convert gradient w.r.t. SIGMA to gradient w.r.t. log-Cholesky parameters
% Uses the fact that d_SIGMA/d_L = L_ij -> 2*L if i=j (diagonal), L if i≠j (off-diagonal)

M = size(L, 1);
grad_L_vec = zeros(size(L_vec));
idx = 1;

for j = 1:M
    for i = j:M
        if i == j
            % Diagonal element: d_SIGMA/d_log(L_ii) = 2 * L_ii * L_ii = 2 * L_ii^2
            % But since we parameterize as log(L_ii), we need: d/d_log(L_ii) = L_ii * d/d_L_ii
            grad_L_vec(idx) = L(i, j) * 2 * grad_SIGMA(i, j);
        else
            % Off-diagonal element: d_SIGMA/d_L_ij = 2 * L_ji (since SIGMA = L*L')
            grad_L_vec(idx) = 2 * grad_SIGMA(i, j);
        end
        idx = idx + 1;
    end
end
end

%% ================================================================
%% LOG-CHOLESKY PARAMETERIZATION HELPER FUNCTIONS (same as before)
%% ================================================================
function L_vec = extract_log_cholesky(L)
M = size(L, 1);
L_vec = [];

for j = 1:M
    for i = j:M
        if i == j
            L_vec = [L_vec; log(max(L(i, j), 1e-8))]; % Log of diagonal
        else
            L_vec = [L_vec; L(i, j)]; % Off-diagonal elements
        end
    end
end
end

function L = reconstruct_log_cholesky(L_vec, M)
L = zeros(M, M);
idx = 1;

for j = 1:M
    for i = j:M
        if i == j
            L(i, j) = exp(L_vec(idx)); % Exp of diagonal ensures positivity
        else
            L(i, j) = L_vec(idx); % Off-diagonal elements
        end
        idx = idx + 1;
    end
end
end

%% ================================================================
%% HELPER FUNCTIONS (same as before but kept for completeness)
%% ================================================================
function V_inv = build_joint_minnesota_precision(M, p, constant, kappa1, kappa2, kappa3, s_squared)
KK = constant + M * p;

% Pre-allocate sparse matrix for efficiency
V_inv = sparse(M * KK, M * KK);

% Build block diagonal structure more efficiently
for eq = 1:M
    start_idx = (eq - 1) * KK + 1;
    end_idx = eq * KK;
    
    V_eq = build_minnesota_precision(eq, M, p, constant, kappa1, kappa2, kappa3, s_squared);
    V_inv(start_idx:end_idx, start_idx:end_idx) = V_eq;
end

% Convert to full matrix for optimization (fminunc works better with full matrices)
V_inv = full(V_inv);
end

function V_inv = build_minnesota_precision(eq_idx, M, p, constant, kappa1, kappa2, kappa3, s_squared)
KK = constant + M * p;

% Pre-allocate diagonal matrix for speed
diag_elements = zeros(KK, 1);
idx = 1;

if constant
    diag_elements(idx) = 1 / kappa3;
    idx = idx + 1;
end

% Vectorized computation of diagonal elements
for lag = 1:p
    lag_factor = 1 / (lag^2);
    for var = 1:M
        if var == eq_idx
            diag_elements(idx) = lag_factor / (kappa1 * s_squared(eq_idx));
        else
            diag_elements(idx) = lag_factor / (kappa2 * s_squared(var));
        end
        idx = idx + 1;
    end
end

V_inv = diag(diag_elements);
end

function s_squared = compute_residual_variances(Y, ar_lags)
[T, M] = size(Y);
s_squared = zeros(M, 1);

% Vectorized computation where possible
for i = 1:M
    y = Y(:, i);
    
    % More efficient lagged matrix construction
    Y_lag = zeros(T - ar_lags, ar_lags);
    for lag = 1:ar_lags
        Y_lag(:, lag) = y(ar_lags + 1 - lag : end - lag);
    end
    
    X_ar = [ones(T - ar_lags, 1), Y_lag];
    y_ar = y(ar_lags + 1 : end);
    
    % Use pre-computed XtX for efficiency
    XtX_ar = X_ar' * X_ar;
    Xty_ar = X_ar' * y_ar;
    
    beta_ar = XtX_ar \ Xty_ar;
    resid = y_ar - X_ar * beta_ar;
    
    s_squared(i) = (resid' * resid) / (length(resid) - ar_lags - 1);
end
end

function beta = calibrate_loss_scale_fast(Y, X, loss_fun, XtX, XtY)
% Optimized version using pre-computed matrices with analytical derivatives
[T, M] = size(Y);

% Initial OLS estimate using pre-computed matrices
A_hat = XtX \ XtY;
E = Y - X * A_hat;

% Compute derivatives based on loss function (vectorized)
E_vec = E(:);
switch lower(loss_fun)
    case 'ols'
        grad = E_vec;
        hess = ones(T*M, 1);
        
    case 'lad'
        grad = sign(E_vec);
        hess = zeros(T*M, 1);
        
    case 'huber'
        delta = 1.345;
        grad = min(max(E_vec, -delta), delta);
        hess = double(abs(E_vec) <= delta);
        
    case 'studentt'
        nu = 4;
        grad = (nu + 1) * E_vec ./ (nu + E_vec.^2);
        hess = (nu + 1) * (nu - E_vec.^2) ./ (nu + E_vec.^2).^2;
        
    otherwise
        grad = E_vec;
        hess = ones(T*M, 1);
end

% Optimized trace computation
X_rep = repmat(X, M, 1);
X_squared_sum = sum(X_rep.^2, 2);
I_trace = sum(grad.^2 .* X_squared_sum);
J_trace = sum(hess .* X_squared_sum);

if J_trace > 1e-12
    beta = I_trace / J_trace;
else
    beta = 1.0;
end

beta = max(min(beta, 10), 0.1);
end

% Optimized prepare_BVAR_matrices function
function [Y_new, X, M, T, KK, names] = prepare_BVAR_matrices(Y, p, constant)
[T_orig, M] = size(Y);
T = T_orig - p;

% Pre-allocate X matrix
if constant
    KK = 1 + M * p;
    X = zeros(T, KK);
    X(:, 1) = 1;
    start_col = 2;
else
    KK = M * p;
    X = zeros(T, KK);
    start_col = 1;
end

% Efficient lagged matrix construction
for lag = 1:p
    end_col = start_col + M - 1;
    X(:, start_col:end_col) = Y(p+1-lag:T_orig-lag, :);
    start_col = end_col + 1;
end

Y_new = Y(p+1:end, :);
names = [];
end