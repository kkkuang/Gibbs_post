function [A, SIGMA, elbo_trace] = BVARlossVB(Y_in, p, constant, loss_fun, varargin)
% BVARlossVB_twofamilies - Variational Bayes for BVAR supporting two families
% Implements a robust Variational Bayes algorithm for a BVAR with a general
% (non-Gaussian) loss function. It supports two distinct variational families.
%
% --- SYNTAX ---
%   [A, SIGMA, elbo_trace] = BVARlossVBs(Y, p, constant, loss_fun, ...)
%
% --- METHODOLOGY ---
% This function approximates the intractable posterior p(A, SIGMA | Y) with a
% simpler distribution q(A, SIGMA) by minimizing the KL-divergence between them.
%
% --- VARIATIONAL FAMILIES ---
% 1. 'mean_field': Assumes full factorization, q(A,SIGMA) = q(SIGMA) * [Π q(a_i)],
%    where a_i are the coefficients for each equation. This is computationally
%    efficient as it ignores correlations between the coefficients of different
%    equations in the approximation.
% 2. 'full_gaussian': Assumes q(A) is a single, large multivariate Gaussian,
%    q(A) = N(vec(A)|mu_A, Sigma_A), which captures correlations between all
%    VAR coefficients. It still assumes q(A,SIGMA) = q(A)q(SIGMA).

% --- OPTIONAL NAME-VALUE PAIRS ---
%   'kappa1','kappa2','kappa3': Minnesota prior hyperparams (defaults large -> weak)
%   'ar_lags': Lags for AR models to estimate residual variances for prior scaling (def: 4)
%   'max_iter': Maximum CAVI iterations (def: 100)
%   'tolerance': Convergence tolerance for the mean parameters (def: 1e-6)
%   'variational_family': 'mean_field' (default) or 'full_gaussian'
%   'compute_elbo': true/false (default false). If true, returns elbo_trace.

% -------------------- Parse Inputs --------------------
parser = inputParser;
addParameter(parser,'kappa1',1000,@isnumeric);
addParameter(parser,'kappa2',1000,@isnumeric);
addParameter(parser,'kappa3',1000,@isnumeric);
addParameter(parser,'ar_lags',4,@isnumeric);
addParameter(parser,'max_iter',1000,@isnumeric);
addParameter(parser,'tolerance',1e-6,@isnumeric);
addParameter(parser,'variational_family','mean_field',@(s)ischar(s) || isstring(s));
addParameter(parser,'compute_elbo',false,@islogical);
parse(parser,varargin{:});
opts = parser.Results;

% -------------------- Prepare Data --------------------
[Y, X, M, T, KK, ~] = prepare_BVAR_matrices(Y_in, p, constant);
if isempty(X) || isempty(Y) || T < p+1
    warning('Insufficient data. Returning NaNs.');
    A = nan(M,KK); SIGMA = nan(M,M); elbo_trace = nan(0,1); return;
end

% --- Initialization ---
A = nan(M,KK); SIGMA = nan(M,M); elbo_trace = [];

% Priors and pre-calculations
s_squared = compute_residual_variances_local(Y, opts.ar_lags);
Psi_0 = diag(s_squared); v_0 = M + 2;               % IW prior for SIGMA
XtX = X' * X;                                       % KK x KK

% Initialize q(A)
A_mean = (X'*X + 1e-6*eye(KK)) \ (X'*Y);           % KK x M (per-eq columns)
if strcmpi(opts.variational_family,'mean_field')
    A_cov = cell(M,1); for i=1:M, A_cov{i} = eye(KK); end
    mu_A = []; Sigma_A = [];
elseif strcmpi(opts.variational_family,'full_gaussian')
    mu_A = reshape(A_mean, KK*M, 1);                % vec(A_mean)
    Sigma_A = eye(KK*M);
else
    error('Unknown variational_family: %s', opts.variational_family);
end

v_post = v_0 + T;
A_mean_prev = A_mean;
if opts.compute_elbo, elbo_trace = zeros(opts.max_iter,1); end

fprintf('VB (%s) with loss=%s: iterating...\n', opts.variational_family, loss_fun);
tic;
try
    for iter = 1:opts.max_iter
        % ---------- 1) Update q(A) ----------
        if strcmpi(opts.variational_family,'mean_field')
            for i = 1:M
                V_prior_inv = build_minnesota_precision_local(i, M, p, constant, opts.kappa1, opts.kappa2, opts.kappa3, s_squared);
                [g_i, H_i] = get_loss_derivatives_local(Y(:,i), X, A_mean(:,i), loss_fun);
                V_post_inv = H_i + V_prior_inv + 1e-8*eye(KK);
                V_post = inv(V_post_inv);
                % Damped Newton step
                step = V_post * (H_i * A_mean(:,i) - g_i) - A_mean(:,i);
                maxstep = 1.0; s = norm(step);
                if s > maxstep, step = step * (maxstep / s); end
                A_mean(:,i) = A_mean(:,i) + 0.5 * step;
                A_cov{i} = V_post;
            end
        else % full_gaussian
            g_big = zeros(KK*M,1);
            H_big = zeros(KK*M, KK*M);
            Vprior_big = zeros(KK*M, KK*M);
            for i = 1:M
                Vp_inv_i = build_minnesota_precision_local(i, M, p, constant, opts.kappa1, opts.kappa2, opts.kappa3, s_squared);
                [g_i, H_i] = get_loss_derivatives_local(Y(:,i), X, A_mean(:,i), loss_fun);
                idx = (i-1)*KK + (1:KK);
                g_big(idx) = g_i;
                H_big(idx, idx) = H_i;
                Vprior_big(idx, idx) = Vp_inv_i;
            end
            V_post_inv_big = H_big + Vprior_big + 1e-8*eye(KK*M);
            Sigma_A = inv(V_post_inv_big);                      % (KK*M)x(KK*M)
            step_big = Sigma_A * ( H_big * mu_A - g_big ) - mu_A;
            maxstep = 5.0; s = norm(step_big);
            if s > maxstep, step_big = step_big * (maxstep / s); end
            mu_A = mu_A + 0.5 * step_big;                        % damping 0.5
            A_mean = reshape(mu_A, KK, M);
        end

        % ---------- 2) Update q(SIGMA) ----------
        E = Y - X * A_mean;                 % T x M
        E_res_sq = E' * E;                  % M x M
        if strcmpi(opts.variational_family,'mean_field')
            for i = 1:M
                corr = trace(A_cov{i} * XtX);    % scalar
                E_res_sq(i,i) = E_res_sq(i,i) + corr;
            end
        else
            for i = 1:M
                idx = (i-1)*KK + (1:KK);
                Sigma_ai = Sigma_A(idx, idx);
                corr = trace(Sigma_ai * XtX);
                E_res_sq(i,i) = E_res_sq(i,i) + corr;
            end
        end
        S_post = Psi_0 + E_res_sq;

        % ---------- 3) Optional ELBO ----------
        if opts.compute_elbo
            elbo_trace(iter) = compute_elbo_approx(Y, X, loss_fun, A_mean, A_cov, S_post, v_post, M, p, constant, opts, s_squared, mu_A, Sigma_A);
        end

        % ---------- 4) Convergence ----------
        if max(abs(A_mean(:) - A_mean_prev(:))) < opts.tolerance, break; end
        A_mean_prev = A_mean;
    end

    % ---------- Outputs ----------
    if strcmpi(opts.variational_family,'full_gaussian')
        A = reshape(mu_A, KK, M);   % KK x M
    else
        A = A_mean;                 % KK x M
    end
    SIGMA = S_post / (v_post - M - 1);
    if opts.compute_elbo, elbo_trace = elbo_trace(1:iter); end

catch ME
    warning('VB failed: %s\n%s', ME.message, ME.getReport());
    if opts.compute_elbo && isempty(elbo_trace), elbo_trace = nan(0,1); end
    return;
end
fprintf('VB completed in %.2f s after %d iters.\n', toc, iter);
end

%% ---------------- Helper Local Functions ----------------
function elbo = compute_elbo_approx(Y, X, loss_fun, A_mean, A_cov, S_post, v_post, M, p, constant, opts, s_squared, mu_A, Sigma_A)
    KK = size(A_mean,1);
    loss_mean = 0; trace_term = 0;
    for i = 1:M
        [~, H_i] = get_loss_derivatives_local(Y(:,i), X, A_mean(:,i), loss_fun);
        loss_i = compute_loss_value_local(Y(:,i) - X*A_mean(:,i), loss_fun);
        loss_mean = loss_mean + loss_i;
        if strcmpi(opts.variational_family,'mean_field')
            trace_term = trace_term + 0.5 * trace(H_i * A_cov{i});
        else
            Sigma_ai = Sigma_A((i-1)*KK+(1:KK), (i-1)*KK+(1:KK));
            trace_term = trace_term + 0.5 * trace(H_i * Sigma_ai);
        end
    end
    E_loglik = -(loss_mean + trace_term);

    prior_quad = 0;
    if strcmpi(opts.variational_family,'mean_field')
        for i = 1:M
            Vp_inv = build_minnesota_precision_local(i, M, p, constant, opts.kappa1, opts.kappa2, opts.kappa3, s_squared);
            prior_quad = prior_quad + 0.5 * (A_mean(:,i)' * Vp_inv * A_mean(:,i) + trace(Vp_inv * A_cov{i}));
        end
    else
        Vprior_big = zeros(KK*M);
        for i=1:M
            idx = (i-1)*KK + (1:KK);
            Vprior_big(idx, idx) = build_minnesota_precision_local(i, M, p, constant, opts.kappa1, opts.kappa2, opts.kappa3, s_squared);
        end
        prior_quad = 0.5 * (mu_A' * Vprior_big * mu_A + trace(Vprior_big * Sigma_A));
    end

    if strcmpi(opts.variational_family,'mean_field')
        H_A = 0; for i=1:M, H_A = H_A + 0.5 * logdet2pie(A_cov{i}); end
    else
        H_A = 0.5 * logdet2pie(Sigma_A);
    end

    % SIGMA terms omitted (constant wrt A-params for tracing convergence)
    elbo = E_loglik - prior_quad + H_A;
end

function s = logdet2pie(S)
    [R, p] = chol((S + S')/2);
    if p > 0, s = -inf; else, s = 2*sum(log(diag(R))) + size(S,1)*log(2*pi*exp(1)); end
end

function [g, H] = get_loss_derivatives_local(y, X, a, loss_fun)
    e = y - X*a;
    switch lower(loss_fun)
        case 'ols'
            psi = e; psi_prime = ones(size(e));
        case 'lad'
            psi = sign(e); psi_prime = zeros(size(e));
        case 'huber'
            delta = 1.345; absE = abs(e); is_small = absE <= delta;
            psi = zeros(size(e)); psi(is_small) = e(is_small); psi(~is_small) = delta * sign(e(~is_small));
            psi_prime = double(is_small);
            psi_prime(~is_small) = 1e-6;  % floor to avoid singular H
        case 'studentt'
            nu = 4; psi = (nu+1) * e ./ (nu + e.^2); psi_prime = (nu+1) * (nu - e.^2) ./ (nu + e.^2).^2;
        otherwise
            error('Unsupported loss function %s', loss_fun);
    end
    g = - X' * psi;                                 % KK x 1
    H = X' * (bsxfun(@times, psi_prime, X));        % KK x KK
end

function loss = compute_loss_value_local(e, loss_fun)
    switch lower(loss_fun)
        case 'ols', loss = 0.5 * sum(e.^2);
        case 'lad', loss = sum(abs(e));
        case 'huber'
            delta = 1.345; ae = abs(e); is_small = ae <= delta;
            loss = sum(0.5 * e(is_small).^2) + sum(delta * (ae(~is_small) - 0.5*delta));
        case 'studentt', nu = 4; loss = -sum(log((1 + e.^2/nu).^(-(nu+1)/2)));
        otherwise, error('Unsupported loss');
    end
end

function V_inv = build_minnesota_precision_local(eq_idx, M, p, constant, k1, k2, k3, s2)
    KK = constant + M*p; V_inv = zeros(KK,KK); idx = 1;
    if constant, V_inv(idx,idx) = 1/k3; idx = idx+1; end
    for lag = 1:p
        for var = 1:M
            if var == eq_idx
                prior_var = k1 / (lag^2);
            else
                prior_var = (k2 * s2(eq_idx)) / (lag^2 * s2(var));
            end
            V_inv(idx,idx) = 1 / prior_var; idx = idx + 1;
        end
    end
end

function s2 = compute_residual_variances_local(Y, lags)
    [T, M] = size(Y); s2 = zeros(M,1);
    for i = 1:M
        if lags >= T, s2(i) = var(Y(:,i)); continue; end
        X_lags = lagmatrix(Y(:,i), 1:lags);
        X_full = [ones(T,1), X_lags];
        rows = (lags+1):T; y_ar = Y(rows, i); X_ar = X_full(rows, :);
        if rank(X_ar) < size(X_ar,2)
            s2(i) = var(y_ar);
        else
            b = (X_ar'*X_ar) \ (X_ar'*y_ar);
            r = y_ar - X_ar*b;
            s2(i) = (r'*r) / (length(r) - size(X_ar,2));
        end
    end
end
