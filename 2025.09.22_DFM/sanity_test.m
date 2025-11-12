%% sanity_test_all_VAR_estimators.m
% Comprehensive sanity test for a suite of VAR estimators (OLS, Huber,
% robust sparse, Lasso, BVAR-ALD, BVAR-ALD-L(1/2), BVAR-Huber,
% BVAR-Minn-Huber, BVAR-SSVS-Huber, BVAR-SMC-Huber, BVAR-VB-Huber).
%
% What this script does:
%   1) Simulate a stable VAR(p) with known A_true and SIGMA_true.
%   2) Fit the listed estimators (if their functions are on-path).
%   3) Enforce a common coefficient orientation (A: M x KK) and compute:
%        - In-sample MSE  = mean((Y_reg - X_eval*A') .^ 2) over t,vars
%        - Coefficient RMSE vs A_true (Frobenius / sqrt(#params))
%        - SIGMA_relErr   = ||Sigma_est - Sigma_true||_F / ||Sigma_true||_F
%          (by default uses a consistent plugin estimate from residuals)
%   4) Print a neat comparison table and quick plots.
%
% Notes:
%  • This test is on clean Gaussian data (no outliers) to check plumbing.
%  • Some estimators may require your own helper functions on the path.
%  • If an estimator returns SIGMA, we also compute its "native" Sigma error
%    but the table uses the plugin version for apples-to-apples.

%% ------------------- CONFIG -------------------
seed        = 123;
T           = 300;        % total time points
M           = 5;          % # variables
p           = 2;          % VAR lags
constant    = 1;          % 1 -> include intercept
loss_fun    = 'huber';    % 'ols' or 'huber'

% Estimator settings
vb_family     = 'mean_field';  % 'mean_field' | 'full_gaussian'
vb_max_iter   = 400;
smc_particles = 2000;
smc_moves     = 3;
smc_kappa     = [0.1, 0.2, 100];   % [kappa1 kappa2 kappa3]

% Which estimators to run (must match cases below)
estimator_names = { ...
    'OLS', ...
    'Huber', ...
    'Robust sparse', ...
    'Lasso', ...
    'BVAR-ALD', ...
    'BVAR-ALD-L(1/2)', ...
    'BVAR-Huber', ...
    'BVAR-Minn-Huber', ...
    'BVAR-SSVS-Huber', ...
    'BVAR-SMC-Huber', ...
    'BVAR-VB-Huber' ...
};

%% ------------------- Simulate VAR -------------------
rng(seed);
[A_true_lags, c_true] = make_stable_lags(M, p, 0.45);  % spectral radius <~ 0.45
SIGMA_true = make_spd_cov(M, 0.3);

Y = simulate_var(T, M, p, constant, A_true_lags, c_true, SIGMA_true);
[X_eval, Y_reg, KK] = build_XY(Y, p, constant);
A_true = pack_A_true(A_true_lags, c_true, M, p, constant); % M x KK
T_eff  = size(Y_reg,1);

%% ------------------- Baseline: OLS (always available) -------------------
A_ols = (X_eval \ Y_reg)';
E_ols = Y_reg - X_eval*A_ols';
MSE_ols     = mean(E_ols.^2, 'all');
RMSE_A_ols  = frob_rmse(A_ols, A_true);
SIGMA_ols   = (E_ols' * E_ols) / T_eff;  % plugin, matches cov(E,1)
SIGMAe_ols  = frob_rel_err(SIGMA_ols, SIGMA_true);

%% ------------------- Run all estimators -------------------
E = numel(estimator_names);
A_est      = NaN(M, KK, E);
MSE        = NaN(1, E);
RMSE_A     = NaN(1, E);
SIGMA_relE = NaN(1, E);        % plugin residual covariance error
SIGMA_relE_native = NaN(1, E); % if estimator returns SIGMA, also store

for e = 1:E
    nm = estimator_names{e};
    fprintf('Running %-18s ... ', nm);
    tic;
    try
        switch lower(nm)
            case 'ols'
                Ahat = A_ols; SIGMAhat_native = SIGMA_ols;

            case 'huber'
                must_exist('robust_var');
                [Ahat, SIGMAhat_native] = robust_var(Y, p, constant, 'loss','huber','cov_method','mcd','verbose', false);

            case 'robust sparse'
                must_exist('sparse_robust_admm');
                tau_seq = quantile(abs(Y), [0.5, 0.75, 0.9, 0.95, 1]);
                tau_com = combos(tau_seq);                       
                lambda_seq = logspace(-2, 3, 10);
                Ahat = sparse_robust_admm(Y, p, constant, tau_com, lambda_seq, 1e-5, 1000);
                SIGMAhat_native = [];  % compute plugin below

            case 'lasso'
                must_exist('Lasso_VAR');
                lambda_seq = logspace(-2, 3, 10);
                Ahat = Lasso_VAR(Y, p, constant, lambda_seq, 1e-6, 1000);
                SIGMAhat_native = [];

            case 'bvar-ald'
                must_exist('BVARquantile');
                [Ahat, SIGMAhat_native] = BVARquantile(Y, p, constant, 0.5*ones(1,M), 20000, 'prior_type','noninformative');

            case 'bvar-ald-l(1/2)'
                must_exist('BVARquantile');
                [Ahat, SIGMAhat_native] = BVARquantile(Y, p, constant, 0.5*ones(1,M), 20000);

            case 'bvar-huber'
                must_exist('BVARlossLLB');
                [Ahat, SIGMAhat_native] = BVARlossLLB(Y, p, constant, 'huber', 20000, 'calibrate_w', true);

            case 'bvar-minn-huber'
                must_exist('BVARlossLLB');
                [Ahat, SIGMAhat_native] = BVARlossLLB(Y, p, constant, 'huber', 20000, 'calibrate_w', true, 'kappa1', 0.1, 'kappa2', 0.1);

            case 'bvar-ssvs-huber'
                must_exist('BVARlossLLB_ssvs');
                [Ahat, SIGMAhat_native] = BVARlossLLB_ssvs(Y, p, constant, 'huber', 20000, 'calibrate_w', true);

            case 'bvar-smc-huber'
                must_exist('BVARlossSMC');
                [Ahat, SIGMAhat_native] = BVARlossSMC(Y, p, constant, 'huber', smc_particles, ...
                    'mcmc_steps', smc_moves, 'kappa1', smc_kappa(1), 'kappa2', smc_kappa(2), 'kappa3', smc_kappa(3));

            case 'bvar-vb-huber'
                must_exist('BVARlossVB');
                [Ahat, SIGMAhat_native] = BVARlossVB(Y, p, constant, 'huber', 'variational_family', vb_family, 'max_iter', vb_max_iter);

            otherwise
                error('Estimator "%s" not implemented.', nm);
        end

        % ---- Standardize A shape to MxKK ----
        Ahat = coerce_A_MxKK(Ahat, M, KK, nm);

        % ---- Metrics (plugin Σ for comparability) ----
        Ehat = Y_reg - X_eval*Ahat';
        MSE(e)    = mean(Ehat.^2, 'all');
        RMSE_A(e) = frob_rmse(Ahat, A_true);
        SIGMA_est_plugin = (Ehat' * Ehat) / T_eff;  % plugin
        SIGMA_relE(e) = frob_rel_err(SIGMA_est_plugin, SIGMA_true);

        % ---- If native Σ was returned, record its error too ----
        if ~isempty(SIGMAhat_native)
            SIGMAhat_native = symproj(SIGMAhat_native);
            SIGMA_relE_native(e) = frob_rel_err(SIGMAhat_native, SIGMA_true);
        end

        A_est(:,:,e) = Ahat;
        fprintf('ok (%.2fs)\n', toc);
    catch ME
        fprintf('FAIL (%.2fs)\n', toc);
        warning('Estimator %s failed: %s', nm, ME.message);
    end
end

%% ------------------- Report -------------------
fprintf('\n=== SANITY CHECK: %s loss | T=%d, M=%d, p=%d ===\n', upper(loss_fun), T, M, p);
fmt = '%-18s  MSE: %10.6f   A-RMSE: %9.6f   Σ-relErr(plugin): %9.6f';
for e = 1:E
    if ~isnan(MSE(e))
        line = sprintf(fmt, estimator_names{e}, MSE(e), RMSE_A(e), SIGMA_relE(e));
        if ~isnan(SIGMA_relE_native(e))
            line = sprintf('%s   [Σ-relErr(native): %9.6f]', line, SIGMA_relE_native(e));
        end
        fprintf('%s\n', line);
    else
        fprintf('%-18s  (skipped or failed)\n', estimator_names{e});
    end
end

%% ------------------- Quick plots -------------------
valid = ~isnan(MSE);
labels = estimator_names(valid);
figure('Name','In-sample MSE');
bar(MSE(valid)); set(gca,'XTickLabel',labels,'XTickLabelRotation',45); ylabel('MSE'); grid on;

figure('Name','Coeff RMSE');
bar(RMSE_A(valid)); set(gca,'XTickLabel',labels,'XTickLabelRotation',45); ylabel('RMSE(A)'); grid on;

figure('Name','Sigma rel. error (plugin)');
bar(SIGMA_relE(valid)); set(gca,'XTickLabel',labels,'XTickLabelRotation',45); ylabel('||Σ-Σ*||_F / ||Σ*||_F'); grid on;

%% ------------------- Helper functions -------------------
function A = coerce_A_MxKK(A, M, KK, nm)
    sz = size(A);
    if numel(sz) ~= 2
        error('%s returned non-2D A of size %s', nm, mat2str(sz));
    end
    if isequal(sz, [M, KK])
        return;
    elseif isequal(sz, [KK, M])
        A = A';
    else
        error('%s returned A of size %s (expected %dx%d or %dx%d)', nm, mat2str(sz), M, KK, KK, M);
    end
end

function r = frob_rmse(Ahat, Atrue)
    r = norm(Ahat - Atrue, 'fro') / sqrt(numel(Atrue));
end

function r = frob_rel_err(S, Strue)
    S = symproj(S); Strue = symproj(Strue);
    r = norm(S - Strue, 'fro') / max(1e-12, norm(Strue, 'fro'));
end

function S = symproj(S)
    S = (S + S')/2;  % enforce symmetry
end

function must_exist(fname)
    if exist(fname, 'file') ~= 2 && exist(fname, 'builtin') ~= 5
        error('Required function %s is not on the MATLAB path.', fname);
    end
end

%% ====== Minimal VAR helpers (local, independent of your toolbox) ======
function [A_lags, c] = make_stable_lags(M, p, target_radius)
    A_lags = cell(p,1);
    for L=1:p
        G = randn(M,M) * 0.3; G = (G + G')/2; A_lags{L} = 0.3 * G;
    end
    C = companion_from_lags(A_lags);
    r = max(abs(eig(C))); scale = (target_radius / max(r, 1e-8));
    for L=1:p, A_lags{L} = scale * A_lags{L}; end
    c = zeros(M,1);
end

function C = companion_from_lags(A_lags)
    p = numel(A_lags); M = size(A_lags{1},1);
    C = zeros(M*p);
    for L=1:p, C(1:M, (L-1)*M+1:L*M) = A_lags{L}; end
    if p>1, C(M+1:end, 1:(p-1)*M) = eye((p-1)*M); end
end

function S = make_spd_cov(M, offdiag)
    R = randn(M); Q = orth(R); d = linspace(1, 2, M);
    S = Q*diag(d)*Q'; D = diag(1./sqrt(diag(S))); S = D*S*D; 
    S = (1-offdiag)*eye(M) + offdiag*S; S = (S+S')/2; 
    [V,Dl] = eig(S); Dl = max(Dl, 1e-6*eye(M)); S = V*Dl*V';
end

function Y = simulate_var(T, M, p, constant, A_lags, c, SIGMA)
    Y = zeros(T, M); e = mvnrnd(zeros(M,1), SIGMA, T);
    for t = p+1:T
        ylag = zeros(M,1);
        for L=1:p, ylag = ylag + A_lags{L} * Y(t-L, :)'; end
        if constant, y_t = c + ylag + e(t,:)'; else, y_t = ylag + e(t,:)'; end
        Y(t,:) = y_t';
    end
end

function [X, Yreg, KK] = build_XY(Y, p, constant)
    [T, M] = size(Y); T_eff = T - p; KK = constant + M*p;
    X = zeros(T_eff, KK); Yreg = Y(p+1:end, :); col = 1;
    if constant, X(:,col) = 1; col = col+1; end
    for L=1:p, X(:, col:col+M-1) = Y(p+1-L:T-L, :); col = col + M; end
end

function A = pack_A_true(A_lags, c, M, p, constant)
    KK = constant + M*p; A = zeros(M, KK); col = 1;
    if constant, A(:,col) = c(:); col = col+1; end
    for L=1:p, A(:, col:col+M-1) = A_lags{L}; col = col + M; end
end
