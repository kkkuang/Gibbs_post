function [A, SIGMA, A_samples, SIGMA_samples] = BVARlossLLB_ssvs(Y, p, constant, loss_fun, num_samples, varargin)

%% ================================================================
% BVARlossLLB - Bayesian VAR with Loss-Likelihood Bootstrap
% Implements Lyddon et al. (2019) loss-likelihood bootstrap for BVAR
% with different loss functions and SSVS priors
%% ================================================================
%  INPUT
%    Y            TxM matrix of endogenous variables  
%    p            Number of lags
%    constant     1 if intercept, 0 otherwise
%    loss_fun     Loss function: 'ols', 'lms', 'lad', 'huber', 'studentt'
%    num_samples  Number of bootstrap samples
%    varargin     Optional: 'calibrate_w', true/false (default: true)
%               NEW:       'eta', scalar (default 1) — fractional-likelihood tempering
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
addParameter(parser, 'eta', 1.0, @isnumeric);      % NEW
parse(parser, varargin{:});
calibrate_w = parser.Results.calibrate_w;
use_parallel = parser.Results.parallel;
eta          = max(0, parser.Results.eta);

% Prepare data
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y, p, constant);

% Prior settings
tau0 = sqrt(0.1); 
tau1 = sqrt(4); 
pi0 = 0.1;
Psi_0 = eye(M); 
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

fprintf('Running loss-likelihood bootstrap with %s loss (SSVS prior, η=%.3g)...\n', loss_fun, eta);
tic;

%% Main Bootstrap Loop
if use_parallel
    parfor i = 1:num_samples
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = single_bootstrap_sample( ...
            Y, X, M, T, KK, beta, loss_fun, tau0, tau1, pi0, Psi_0, v_0, eta);
    end
else
    for i = 1:num_samples
        if mod(i, 100) == 0, fprintf('Sample %d/%d\n', i, num_samples); end
        [A_samples(i,:,:), SIGMA_samples(i,:,:)] = single_bootstrap_sample( ...
            Y, X, M, T, KK, beta, loss_fun, tau0, tau1, pi0, Psi_0, v_0, eta);
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
function [A_sample, SIGMA_sample] = single_bootstrap_sample(Y, X, M, T, KK, beta, loss_fun, tau0, tau1, pi0, Psi_0, v_0, eta)

% 1. Generate Dirichlet weights
weights = gamrnd(ones(T, 1), 1);
weights = weights / sum(weights) * (eta*T);   % NEW: tempering

% 2. Fast weighted OLS starting point
W = spdiags(weights, 0, T, T);
A_init = (X' * W * X) \ (X' * W * Y);  % KK x M

% 3. Sample SSVS indicators
GAMMA = zeros(KK, M);
for i = 1:M
    for j = 1:KK
        l_0 = -0.5 * log(2*pi*tau0^2) - 0.5 * (A_init(j,i)/tau0)^2;
        l_1 = -0.5 * log(2*pi*tau1^2) - 0.5 * (A_init(j,i)/tau1)^2;
        pip = 1 / (1 + ((1-pi0)/pi0) * exp(l_0 - l_1));
        GAMMA(j,i) = rand < pip;
    end
end

% 4. Solve robust problem
A_opt = solve_robust_problem(Y, X, weights, A_init, GAMMA, tau0, tau1, beta, loss_fun);

% 5. Sample SIGMA
E = Y - X * A_opt;
S_post = Psi_0 + (E .* weights)' * E;
df_post = v_0 + sum(weights);          % NEW: df matches tempering
try
    SIGMA_sample = iwishrnd(S_post, df_post);
catch
    SIGMA_sample = S_post / max(df_post - M - 1, 1);
end

A_sample = A_opt';  % Convert to M x KK
end
 
%% ================================================================
%% Solve Robust Problem
%% ================================================================
function A_opt = solve_robust_problem(Y, X, weights, A_init, GAMMA, tau0, tau1, beta, loss_fun)
[T, M] = size(Y);
KK = size(X, 2);

switch lower(loss_fun)
    case 'ols'
        % Analytical solution for OLS
        A_opt = solve_weighted_ols_analytical(Y, X, weights, GAMMA, tau0, tau1);
        
    case 'lms'
        % True Least Median of Squares
        A_opt = solve_true_lms(Y, X, weights, A_init, GAMMA, tau0, tau1);
        
    otherwise
        % Use IRLS for other robust losses
        A_opt = solve_irls(Y, X, weights, A_init, GAMMA, tau0, tau1, beta, loss_fun);
end
end

%% ================================================================
%% Analytical OLS Solution
%% ================================================================
function A_opt = solve_weighted_ols_analytical(Y, X, weights, GAMMA, tau0, tau1)
[T, M] = size(Y);
KK = size(X, 2);
W = spdiags(weights, 0, T, T);

A_opt = zeros(KK, M);
for i = 1:M
    % Prior precision matrix for equation i
    V_inv = diag(GAMMA(:,i) / tau1^2 + (1-GAMMA(:,i)) / tau0^2);
    
    % Posterior computation
    Sigma_inv = X' * W * X + V_inv;
    mu = Sigma_inv \ (X' * W * Y(:,i));
    A_opt(:,i) = mu;
end
end

%% ================================================================
%% True Least Median of Squares Implementation
%% ================================================================
function A_opt = solve_true_lms(Y, X, weights, A_init, GAMMA, tau0, tau1)
[T, M] = size(Y);
KK = size(X, 2);

A_opt = zeros(KK, M);

% Solve equation by equation
for eq = 1:M
    % Prior precision for this equation
    V_inv = diag(GAMMA(:,eq) / tau1^2 + (1-GAMMA(:,eq)) / tau0^2);
    
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
%% Weighted Median Loss Function
%% ================================================================
function loss = weighted_median_loss(beta, y, X, weights)
% Compute weighted median of squared residuals
% For weighted median, we need to sort residuals and find the median

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
function A_opt = solve_irls(Y, X, weights, A_init, GAMMA, tau0, tau1, beta, loss_fun)
[T, M] = size(Y);
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
        V_inv = diag(GAMMA(:,eq) / tau1^2 + (1-GAMMA(:,eq)) / tau0^2);
        
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
%% Fast Calibration (No longer needed for LMS but kept for other losses)
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