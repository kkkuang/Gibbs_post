function [A, SIGMA, A_samples, SIGMA_samples] = BVARlossSMC(Y_in, p, constant, loss_fun, num_particles, varargin)
% BVARlossSMC - Bayesian VAR with Sequential Monte Carlo (tempered SMC)
% Supports general loss functions (ols, lad, huber, studentt).
%
% ORIENTATION: A is returned as (KK x M), where KK = constant + M*p, M=#vars.
%              This matches the regression form Y ~ X*A (i.e., Y = X*A + E).
%              If your downstream code expects K×n with K rows, transpose A.
%
% Inputs:
%   Y_in           : T x M data
%   p              : # lags
%   constant       : 1/0 include intercept
%   loss_fun       : 'ols' | 'lad' | 'huber' | 'studentt'
%   num_particles  : N particles
%
% Name-value options:
%   'kappa1', 'kappa2', 'kappa3' : Minnesota prior (defaults 1000,1000,1000)
%   'ar_lags'        : AR lags to scale Minnesota (default 4)
%   'ess_threshold'  : target ESS/N per tempering stage (default 0.7)
%   'mcmc_steps'     : RWMH steps per particle per stage (default 3)
%   'init_scale'     : prior-centered init covariance scale (default 0.1)
%   'use_ols_init'   : mix OLS-centered particles (default true)
%   'init_ols_scale' : OLS-centered init covariance scale (default 0.5)
%   'max_gamma_step' : max ∆γ per stage (default 0.2)
%   'loss_scale'     : normalize loss; if empty uses T_eff*M (default [])
%
% Outputs:
%   A              : KK x M posterior mean of coefficients
%   SIGMA          : M x M posterior mean covariance (plugin IW mean)
%   A_samples      : N x M x KK particle coefficients (for inspection)
%   SIGMA_samples  : []  (not drawn here to keep it fast; see note below)

% ---- Parse options ----
parser = inputParser;
addParameter(parser, 'kappa1', 1000,  @isnumeric);
addParameter(parser, 'kappa2', 1000,  @isnumeric);
addParameter(parser, 'kappa3', 1000,  @isnumeric);
addParameter(parser, 'ar_lags', 4, @isnumeric);
addParameter(parser, 'ess_threshold', 0.7, @isnumeric);
addParameter(parser, 'mcmc_steps', 3, @isnumeric);
addParameter(parser, 'init_scale', 0.1, @isnumeric);
addParameter(parser, 'use_ols_init', true, @islogical);
addParameter(parser, 'init_ols_scale', 0.5, @isnumeric);
addParameter(parser, 'max_gamma_step', 0.2, @isnumeric);
addParameter(parser, 'loss_scale', [], @(x) isempty(x) || isnumeric(x));
parse(parser, varargin{:});
opts = parser.Results;

N = num_particles;

% ---- Prepare data & prior scaling ----
[Y, X, M, T_eff, KK, ~] = prepare_BVAR_matrices(Y_in, p, constant);
if isempty(X) || isempty(Y) || T_eff < 1
    warning('Not enough observations to estimate the model. Exiting.');
    A = nan(KK,M); SIGMA = nan(M,M); A_samples = nan; SIGMA_samples = [];
    return;
end
if isempty(opts.loss_scale), opts.loss_scale = T_eff * M; end

fprintf('SMC: loss=%s, N=%d, ESS target=%.2f, init scales: prior=%.3f, OLS=%.3f\n', ...
    loss_fun, N, opts.ess_threshold, opts.init_scale, opts.init_ols_scale);
tic;

% Default outputs (in case of early exit)
A = nan(KK,M); SIGMA = nan(M,M);
A_samples = nan(N,M,KK); SIGMA_samples = [];

try
    % Residual variances for Minnesota scaling
    s2 = compute_residual_variances_local(Y, opts.ar_lags);   % Mx1
    Psi_0 = diag(s2);
    v_0 = M + 2;

    % Minnesota precision & covariance per equation
    V_prior_inv = zeros(KK, KK, M);
    V_prior = zeros(KK, KK, M);
    for i = 1:M
        V_prior_inv(:,:,i) = build_minnesota_precision_local(i, M, p, constant, ...
            opts.kappa1, opts.kappa2, opts.kappa3, s2);
        V_prior(:,:,i) = inv(V_prior_inv(:,:,i) + 1e-8 * eye(KK));
    end

    % OLS-centered stats
    XtX = X' * X;
    XtX_inv = inv(XtX + 1e-8 * eye(KK));
    A_ols_cols = XtX_inv * (X' * Y);   % KK x M

    % ---- Initialize particles: particles_A is KK x M x N ----
    particles_A = zeros(KK, M, N);
    n_ols   = opts.use_ols_init * floor(0.5 * N);
    n_prior = N - n_ols;

    % prior-centered draws
    if n_prior > 0
        for i = 1:M
            tmp = mvnrnd(zeros(1,KK), opts.init_scale * V_prior(:,:,i), n_prior)'; % KK x n_prior
            particles_A(:, i, 1:n_prior) = reshape(tmp, [KK, 1, n_prior]);
        end
    end

    % OLS-centered draws
    if n_ols > 0
        for i = 1:M
            mu_i = A_ols_cols(:, i)';                                          % 1 x KK
            tmp = mvnrnd(mu_i, opts.init_ols_scale * XtX_inv, n_ols)';         % KK x n_ols
            particles_A(:, i, n_prior+1:N) = reshape(tmp, [KK, 1, n_ols]);
        end
    end

    if any(~isfinite(particles_A(:)))
        warning('SMC initialization produced non-finite values.'); return;
    end

    % ---- Tempering SMC ----
    logw = -log(N) * ones(N,1);   % start equal weights
    gamma_prev = 0;
    max_stages = 5 * ceil(1 / max(1e-8, opts.max_gamma_step));
    stage = 0;

    while (gamma_prev < 1) && (stage < max_stages)
        stage = stage + 1;

        % PRECOMPUTE losses ONCE for this stage (normalized by loss_scale)
        L = calculate_loss_for_all_local(particles_A, Y, X, loss_fun) / opts.loss_scale;  % N x 1
        L(~isfinite(L)) = max(L(isfinite(L)));  % guard

        % Fast γ selection (vectorized ESS over small grid + refine)
        delta_max = min(opts.max_gamma_step, 1 - gamma_prev);
        [gamma_next, logw_new] = choose_gamma_fast(gamma_prev, delta_max, logw, L, opts.ess_threshold);

        % Guards for numerical safety
        if ~isfinite(gamma_next)
            gamma_next = min(gamma_prev + 1e-3, 1);
            logw_new   = -log(N) * ones(N,1);
        end
        if gamma_next <= gamma_prev + 1e-12
            gamma_next = min(gamma_prev + max(1e-3, 0.5*opts.max_gamma_step), 1);
            logw_new   = -log(N) * ones(N,1);
        end

        logw = logw_new;
        w = exp(logw); w = w / sum(w);
        ess = 1 / sum(w.^2);

        % Resample if needed (before move), provided gamma not final
        if ess < opts.ess_threshold * N && gamma_next < 1
            idx = randsample(1:N, N, true, w);
            particles_A = particles_A(:,:,idx);
            logw = -log(N) * ones(N,1);
            w = exp(logw);
        end

        % RWMH move step (few jiggles per particle)
        particles_A_flat = reshape(particles_A, KK*M, N);
        C = cov(particles_A_flat'); C = C + 1e-8 * eye(KK*M);
        d = KK*M; prop_scale = (2.38^2) / max(1, d);

        for i = 1:N
            curr_flat = particles_A_flat(:, i);
            curr = reshape(curr_flat, KK, M);
            for s = 1:opts.mcmc_steps
                prop_flat = mvnrnd(curr_flat', prop_scale * C)';  % d x 1
                prop = reshape(prop_flat, KK, M);
                lp_curr = log_pseudo_posterior_local(curr, Y, X, V_prior_inv, gamma_next, loss_fun, opts.loss_scale);
                lp_prop = log_pseudo_posterior_local(prop,  Y, X, V_prior_inv, gamma_next, loss_fun, opts.loss_scale);
                if isfinite(lp_prop) && isfinite(lp_curr) && (log(rand) < (lp_prop - lp_curr))
                    curr = prop; curr_flat = prop_flat;
                end
            end
            particles_A(:,:,i) = curr;
        end

        gamma_prev = gamma_next;
        % fprintf('[SMC] stage %2d: gamma=%.4f, ESS/N=%.3f\n', stage, gamma_prev, ess/N); % debug
    end

    % ---- Outputs ----
    % Weighted posterior mean of A (KK x M)
    w = exp(logw); w = w / sum(w);
    A_hat = squeeze(sum(reshape(w,1,1,N) .* particles_A, 3));
    if size(A_hat,1) ~= KK, A_hat = A_hat.'; end  % guard
    A = A_hat;

    % Particle dump (N x M x KK) for inspection if needed
    A_samples = permute(particles_A, [3, 2, 1]);

    % Plugin posterior mean of SIGMA (comparable to OLS/VB sanity metrics)
    v_post = v_0 + T_eff;
    E_hat = Y - X * A_hat;                    % T_eff x M
    S_post = Psi_0 + E_hat' * E_hat;          % M x M
    SIGMA = S_post / (v_post - M - 1);
    SIGMA = (SIGMA + SIGMA')/2 + 1e-8*eye(M); % symmetrize/ridge

    % We skip drawing SIGMA_samples for speed; set empty to keep signature
    SIGMA_samples = [];

catch ME
    warning('SMC failed: %s\n%s', ME.message, ME.getReport());
    return;
end

fprintf('SMC Completed in %.2f seconds\n', toc);
end

%% ===================== Helper functions =====================

function [gamma_next, logw_new] = choose_gamma_fast(gamma_prev, delta_max, logw_prev, L, ess_target)
% Fast γ chooser: precompute losses L; evaluate ESS(δ) on a small grid, refine locally.
% Inputs:
%   gamma_prev : current tempering level
%   delta_max  : max increment allowed
%   logw_prev  : N×1 normalized log-weights (sum exp(logw_prev)=1)
%   L          : N×1 per-particle normalized losses  (>=0)
%   ess_target : target ESS fraction in (0,1]
%
% Outputs:
%   gamma_next : next tempering level
%   logw_new   : updated normalized log-weights at gamma_next

    N = numel(L);
    w_prev = exp(logw_prev);
    w_prev = w_prev / sum(w_prev);
    w2_prev = w_prev.^2;

    % coarse grid on δ ∈ (0, delta_max]
    G1 = 8;
    if delta_max <= 1e-12
        delta = 0;
    else
        d1 = linspace(1e-8, delta_max, G1);      % avoid exactly 0
        % ESS(δ) = (Σ w_i e^{-δ L_i})^2 / (Σ w_i^2 e^{-2δ L_i})
        e1 = exp(-L * d1);                       % N x G1
        Z1 = w_prev.'  * e1;                     % 1 x G1
        Z2 = w2_prev.' * (e1.^2);                % 1 x G1
        ESS = (Z1.^2) ./ max(Z2, realmin);

        target = ess_target * N;
        ok = find(ESS >= target, 1, 'last');
        if isempty(ok)
            delta = min(delta_max, 1e-3);        % take tiny safe step
        elseif ok == G1
            delta = d1(end);
        else
            % two bisection refinements between d1(ok) and d1(ok+1)
            lo = d1(ok); hi = d1(ok+1);
            for it = 1:2
                mid = 0.5*(lo+hi);
                e  = exp(-L * mid);
                z1 = w_prev.'  * e;
                z2 = w2_prev.' * (e.^2);
                ess_mid = (z1^2) / max(z2, realmin);
                if ess_mid >= target, lo = mid; else, hi = mid; end
            end
            delta = lo;
        end
    end

    gamma_next = min(1, gamma_prev + delta);

    % update weights at gamma_next
    e  = exp(-L * delta);
    z1 = w_prev.' * e;
    if ~isfinite(z1) || z1 <= 0
        w_new = ones(N,1)/N;
    else
        w_new = (w_prev .* e) / z1;
        if any(~isfinite(w_new)), w_new = ones(N,1)/N; end
    end
    logw_new = log(w_new + realmin);
end

function R_n = calculate_loss_for_all_local(p_A, Y, X, lf)
    N = size(p_A,3);
    R_n = zeros(N,1);
    for i=1:N
        E = Y - X * p_A(:,:,i);
        R_n(i) = compute_loss_value_local(E, lf);
    end
end

function loss = compute_loss_value_local(E, lf)
    switch lower(lf)
        case 'ols'
            loss = 0.5 * sum(E(:).^2);
        case 'lad'
            loss = sum(abs(E(:)));
        case 'huber'
            d = 1.345;
            aE = abs(E); iS = (aE <= d);
            loss = sum(0.5 * E(iS).^2) + sum(d * (aE(~iS) - 0.5*d));
        case 'studentt'
            nu = 4;
            loss = -sum(log((1 + E(:).^2/nu).^(-(nu+1)/2)));
        otherwise
            error('Unsupported loss');
    end
end

function log_p = log_pseudo_posterior_local(A, Y, X, V_inv, g, lf, loss_scale)
    % log prior
    log_prior = 0;
    [~, M] = size(A);
    for i = 1:M
        log_prior = log_prior - 0.5 * A(:,i)' * V_inv(:,:,i) * A(:,i);
    end
    % tempered loss-likelihood
    log_like = - g * ( compute_loss_value_local(Y - X*A, lf) / loss_scale );
    log_p = log_prior + log_like;
end

function V_inv = build_minnesota_precision_local(eq_idx, M, p, constant, k1, k2, k3, s2)
    KK = constant + M*p;
    V_inv = zeros(KK,KK);
    idx = 1;
    if constant
        V_inv(idx,idx) = 1/k3; idx = idx+1;
    end
    for lg = 1:p
        for v = 1:M
            if v == eq_idx
                prior_var = k1 / (lg^2);
            else
                prior_var = (k2 * s2(eq_idx)) / (lg^2 * s2(v));
            end
            V_inv(idx,idx) = 1 / prior_var;
            idx = idx + 1;
        end
    end
end

function s2 = compute_residual_variances_local(Y, lags)
    [T, M] = size(Y);
    s2 = zeros(M, 1);
    for i = 1:M
        if lags >= T
            s2(i) = var(Y(:,i)); continue;
        end
        X_lags = lagmatrix(Y(:,i), 1:lags);
        X_full = [ones(T,1), X_lags];
        y_ar = Y(lags+1:end, i);
        X_ar = X_full(lags+1:end, :);
        if rank(X_ar) < size(X_ar,2)
            s2(i) = var(y_ar);
        else
            b = (X_ar'*X_ar) \ (X_ar'*y_ar);
            r = y_ar - X_ar*b;
            s2(i) = (r'*r) / max(1, (length(r) - size(X_ar,2)));
        end
    end
end
