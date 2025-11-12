function [A, SIGMA, A_samples, SIGMA_samples] = BVARlossLLB(Y, p, constant, loss_fun, num_samples, varargin)

%% ================================================================
% BVARlossLLB - Bayesian VAR with Loss-Likelihood Bootstrap
% Implements Lyddon et al. (2019) loss-likelihood bootstrap for BVAR
% with different loss functions and Minnesota priors
%% ================================================================
%  INPUT
%    Y            TxM matrix of endogenous variables  
%    p            Number of lags
%    constant     1 if intercept, 0 otherwise
%    loss_fun     Loss function: 'ols', 'lms', 'lad', 'huber', 'studentt'
%    num_samples  Number of bootstrap samples
%    varargin     Optional: 'calibrate_w', true/false (default: true)
%                          'kappa1', scalar (default: 1000, noninformative)
%                          'kappa2', scalar (default: 1000, noninformative) 
%                          'kappa3', scalar (default: 1000, noninformative)
%                          'ar_lags', scalar (default: 4, for residual variance)
%               NEW:       'eta', scalar in (0,∞) (default: 1) — fractional likelihood tempering.
%                          Implemented by scaling the Dirichlet weights so
%                          sum(weights)=eta*T (and using v0+sum(weights) in Σ draw).
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
addParameter(parser, 'parallel', false, @islogical);
addParameter(parser, 'kappa1', 1000, @isnumeric);    % Own lag shrinkage
addParameter(parser, 'kappa2', 1000, @isnumeric);    % Other lag shrinkage  
addParameter(parser, 'kappa3', 1000, @isnumeric);    % Intercept shrinkage
addParameter(parser, 'ar_lags', 4, @isnumeric);      % AR lags for residual variance
addParameter(parser, 'eta', 1.0, @isnumeric);        % NEW: likelihood tempering
parse(parser, varargin{:});
calibrate_w = parser.Results.calibrate_w;
use_parallel = parser.Results.parallel;
kappa1 = parser.Results.kappa1;
kappa2 = parser.Results.kappa2;
kappa3 = parser.Results.kappa3;
ar_lags = parser.Results.ar_lags;
eta      = max(0, parser.Results.eta);               % guard

% Prepare data
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y, p, constant);

% Compute residual variances for Minnesota prior
s_squared = compute_residual_variances(Y, ar_lags);

% Prior settings for SIGMA - inverse Wishart
Psi_0 = diag(s_squared); 
v_0 = M + 2;

% Calibration
if calibrate_w && ~strcmpi(loss_fun, 'lms')
    beta = calibrate_loss_scale_fast(Y, X, loss_fun);
    fprintf('Calibrated loss scale beta = %.4f\n', beta);
else
    beta = 1.0;
    if strcmpi(loss_fun, 'lms')
        fprintf('Using beta = 1.0 for LMS (median is scale-invariant)\n');
    end
end

% Pre-allocate
A_samples = zeros(num_samples, M, KK);
SIGMA_samples = zeros(num_samples, M, M);

fprintf('Running loss-likelihood bootstrap with %s loss and Minnesota prior (κ1=%.1f, κ2=%.1f, η=%.3g)...\n', ...
        loss_fun, kappa1, kappa2, eta);
tic;

%% Main Bootstrap Loop
if use_parallel
    parfor i = 1:num_samples
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = single_bootstrap_sample( ...
            Y, X, M, T, KK, p, constant, beta, loss_fun, kappa1, kappa2, kappa3, ...
            s_squared, Psi_0, v_0, eta);
    end
else
    for i = 1:num_samples
        if mod(i, 1000) == 0, fprintf('Sample %d/%d\n', i, num_samples); end
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = single_bootstrap_sample( ...
            Y, X, M, T, KK, p, constant, beta, loss_fun, kappa1, kappa2, kappa3, ...
            s_squared, Psi_0, v_0, eta);
    end
end

fprintf('Completed in %.2f seconds\n', toc);

% Posterior means
A = squeeze(mean(A_samples, 1))';      % KK x M
SIGMA = squeeze(mean(SIGMA_samples, 1)); % M x M
end

%% ================================================================
%% Single Bootstrap Sample
%% ================================================================
function [A_sample, SIGMA_sample] = single_bootstrap_sample(Y, X, M, T, KK, p, constant, ...
    beta, loss_fun, kappa1, kappa2, kappa3, s_squared, Psi_0, v_0, eta)

% 1. Generate Dirichlet weights
weights = gamrnd(ones(T, 1), 1);
weights = weights / sum(weights) * (eta*T);   % NEW: temper → effective sample size ηT

% 2. Fast weighted OLS starting point
W = spdiags(weights, 0, T, T);
A_init = (X' * W * X) \ (X' * W * Y);  % KK x M

% 3. Solve robust problem with Minnesota prior
A_opt = solve_robust_problem(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, ...
                            s_squared, beta, loss_fun);

% 4. Sample SIGMA
E = Y - X * A_opt;
S_post = Psi_0 + (E .* weights)' * E;
df_post = v_0 + sum(weights);          % NEW: df matches tempered mass
try
    SIGMA_sample = iwishrnd(S_post, df_post);
catch
    SIGMA_sample = S_post / max(df_post - M - 1, 1); % safe fallback
end

A_sample = A_opt';  % Convert to M x KK
end

%% ================================================================
%% Solve Robust Problem
%% ================================================================
function A_opt = solve_robust_problem(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, ...
                                     s_squared, beta, loss_fun)
[T, ~] = size(Y);
KK = size(X, 2);

switch lower(loss_fun)
    case 'ols'
        % Analytical solution for OLS
        A_opt = solve_weighted_ols_analytical(Y, X, weights, M, p, constant, kappa1, kappa2, kappa3, s_squared);
        
    case 'lms'
        % True Least Median of Squares
        A_opt = solve_true_lms(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, s_squared);
        
    otherwise
        % Use IRLS for other robust losses
        A_opt = solve_irls(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, s_squared, beta, loss_fun);
end
end

%% ================================================================
%% Analytical OLS Solution with Minnesota Prior
%% ================================================================
function A_opt = solve_weighted_ols_analytical(Y, X, weights, M, p, constant, kappa1, kappa2, kappa3, s_squared)
[T, ~] = size(Y);
KK = size(X, 2);
W = spdiags(weights, 0, T, T);

A_opt = zeros(KK, M);

for i = 1:M
    % Build Minnesota prior precision matrix for equation i
    V_inv = build_minnesota_precision(i, M, p, constant, kappa1, kappa2, kappa3, s_squared);
    
    % Posterior computation
    Sigma_inv = X' * W * X + V_inv;
    mu = Sigma_inv \ (X' * W * Y(:,i));
    A_opt(:,i) = mu;
end
end

%% ================================================================
%% True Least Median of Squares Implementation
%% ================================================================
function A_opt = solve_true_lms(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, s_squared)
[T, ~] = size(Y);
KK = size(X, 2);

A_opt = zeros(KK, M);

% Solve equation by equation
for eq = 1:M
    % Prior precision for this equation
    V_inv = build_minnesota_precision(eq, M, p, constant, kappa1, kappa2, kappa3, s_squared);
    
    % Objective function: weighted median of squared residuals + prior penalty
    objective = @(beta) weighted_median_loss(beta, Y(:,eq), X, weights) + ...
                       0.5 * beta' * V_inv * beta;
    
    % Use multiple starting points to avoid local minima
    best_beta = A_init(:,eq);
    best_obj = objective(best_beta);
    
    % Try several starting points
    num_starts = 5;
    for start = 1:num_starts
        if start == 1
            beta0 = A_init(:,eq);
        else
            % Random perturbation of OLS solution
            beta0 = A_init(:,eq) + 0.1 * randn(KK, 1) * std(A_init(:,eq));
        end
        
        % Optimization options
        options = optimset('Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6, ...
                          'MaxIter', 1000, 'MaxFunEvals', 5000);
        
        try
            % Use fminsearch (derivative-free) since median is non-smooth
            [beta_opt, obj_val] = fminsearch(objective, beta0, options);
            
            if obj_val < best_obj
                best_obj = obj_val;
                best_beta = beta_opt;
            end
        catch
            % If optimization fails, continue with next starting point
            continue;
        end
    end
    
    A_opt(:,eq) = best_beta;
end
end

%% ================================================================
%% Build Minnesota Prior Precision Matrix
%% ================================================================
function V_inv = build_minnesota_precision(eq_idx, M, p, constant, kappa1, kappa2, kappa3, s_squared)
% Build precision matrix for equation eq_idx under Minnesota prior
% V_inv is KK x KK where KK = constant + M*p

KK = constant + M * p;
V_inv = zeros(KK, KK);

idx = 1;

% Intercept term
if constant
    V_inv(idx, idx) = 1 / kappa3;  % Prior precision for intercept
    idx = idx + 1;
end

% VAR coefficients
for lag = 1:p
    for var = 1:M
        if var == eq_idx
            % Own lag - less shrinkage
            prior_var = kappa1 / (lag^2 * s_squared(eq_idx));
        else
            % Other variable lag - more shrinkage
            prior_var = kappa2 / (lag^2 * s_squared(var));
        end
        V_inv(idx, idx) = 1 / prior_var;
        idx = idx + 1;
    end
end
end

%% ================================================================
%% Weighted Median Loss Function
%% ================================================================
function loss = weighted_median_loss(beta, y, X, weights)
% Compute weighted median of squared residuals

residuals = y - X * beta;
squared_residuals = residuals.^2;

% For weighted median, sort by residual value and accumulate weights
[sorted_sq_resid, idx] = sort(squared_residuals);
sorted_weights = weights(idx);

% Find weighted median
cumsum_weights = cumsum(sorted_weights);
total_weight = sum(sorted_weights);
median_threshold = total_weight / 2;

% Find the index where cumulative weight exceeds half
median_idx = find(cumsum_weights >= median_threshold, 1, 'first');

if isempty(median_idx)
    median_idx = length(sorted_sq_resid);
end

% Linear interpolation for weighted median if needed
if median_idx > 1 && cumsum_weights(median_idx-1) < median_threshold
    % Interpolate between median_idx-1 and median_idx
    w1 = median_threshold - cumsum_weights(median_idx-1);
    w2 = cumsum_weights(median_idx) - median_threshold;
    if w1 + w2 > 0
        loss = (w2 * sorted_sq_resid(median_idx-1) + w1 * sorted_sq_resid(median_idx)) / (w1 + w2);
    else
        loss = sorted_sq_resid(median_idx);
    end
else
    loss = sorted_sq_resid(median_idx);
end
end

%% ================================================================
%% IRLS for Other Robust Losses
%% ================================================================
function A_opt = solve_irls(Y, X, weights, A_init, M, p, constant, kappa1, kappa2, kappa3, s_squared, beta, loss_fun)
[T, ~] = size(Y);
KK = size(X, 2);

A_current = A_init;

% IRLS iterations
for iter = 1:5
    % Compute residuals
    E = Y - X * A_current;
    
    % Compute robust weights
    rob_weights = compute_robust_weights(E, loss_fun);
    
    % Combine weights
    combined_weights = weights .* rob_weights;
    
    % Update coefficients equation by equation
    for eq = 1:M
        W = spdiags(combined_weights, 0, T, T);
        V_inv = build_minnesota_precision(eq, M, p, constant, kappa1, kappa2, kappa3, s_squared);
        
        Sigma_inv = X' * W * X + V_inv;
        mu = Sigma_inv \ (X' * W * Y(:,eq));
        A_current(:,eq) = mu;
    end
end

A_opt = A_current;
end

%% ================================================================
%% Compute Robust Weights
%% ================================================================
function rob_weights = compute_robust_weights(E, loss_fun)
[T, M] = size(E);

switch lower(loss_fun)
    case 'lad'
        % LAD weights
        r = sqrt(sum(E.^2, 2));
        rob_weights = 1 ./ max(r, 1e-6);
        
    case 'huber'
        % Huber weights
        delta = 1.345;
        r = sqrt(sum(E.^2, 2));
        rob_weights = min(1, delta ./ max(r, 1e-6));
        
    case 'studentt'
        % Student-t weights
        nu = 4;
        r_sq = sum(E.^2, 2);
        rob_weights = (nu + M) ./ (nu + r_sq);
        
    otherwise
        % Default to unit weights
        rob_weights = ones(T, 1);
end
end

%% ================================================================
%% Compute Residual Variances for Minnesota Prior
%% ================================================================
function s_squared = compute_residual_variances(Y, ar_lags)
% Compute residual variances from AR(ar_lags) models for each variable
[T, M] = size(Y);
s_squared = zeros(M, 1);

for i = 1:M
    y = Y(:, i);
    
    % Create lagged matrix for AR model
    Y_lag = zeros(T - ar_lags, ar_lags);
    for lag = 1:ar_lags
        Y_lag(:, lag) = y(ar_lags + 1 - lag : end - lag);
    end
    
    % Add constant
    X_ar = [ones(T - ar_lags, 1), Y_lag];
    y_ar = y(ar_lags + 1 : end);
    
    % OLS estimation
    beta_ar = (X_ar' * X_ar) \ (X_ar' * y_ar);
    resid = y_ar - X_ar * beta_ar;
    
    % Residual variance
    s_squared(i) = (resid' * resid) / (length(resid) - ar_lags - 1);
end
end

%% ================================================================
%% Fast Calibration
%% ================================================================
function beta = calibrate_loss_scale_fast(Y, X, loss_fun)
[T, M] = size(Y);
KK = size(X, 2);

% Initial OLS estimate
A_hat = (X' * X) \ (X' * Y);
E = Y - X * A_hat;

% Compute derivatives based on loss function
switch lower(loss_fun)
    case 'ols'
        grad = E(:);
        hess = ones(T*M, 1);
        
    case 'lad'
        grad = sign(E(:));
        hess = zeros(T*M, 1);
        
    case 'huber'
        delta = 1.345;
        u = E(:);
        grad = min(max(u, -delta), delta);
        hess = double(abs(u) <= delta);
        
    case 'studentt'
        nu = 4;
        u = E(:);
        grad = (nu + 1) * u ./ (nu + u.^2);
        hess = (nu + 1) * (nu - u.^2) ./ (nu + u.^2).^2;
        
    otherwise
        grad = E(:);
        hess = ones(T*M, 1);
end

% Fast trace computation
X_rep = repmat(X, M, 1);
I_trace = sum(grad.^2 .* sum(X_rep.^2, 2));
J_trace = sum(hess .* sum(X_rep.^2, 2));

if J_trace > 1e-12
    beta = I_trace / J_trace;
else
    beta = 1.0;
end

beta = max(min(beta, 10), 0.1);
end

