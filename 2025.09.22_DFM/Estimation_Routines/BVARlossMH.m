function [A,SIGMA] = BVARlossMH(Y,p,constant,loss_fun,num_samples,varargin)
%% ================================================================
% BVARlossMH   Bayesian VAR with Minnesota prior via MH
% loss_fun: 'ols' | 'lms' | 'lad' | 'huber' | 'studentt'
%% ================================================================

% Parse optional inputs
parser = inputParser;
addParameter(parser, 'kappa1', 1000, @isnumeric);
addParameter(parser, 'kappa2', 1000, @isnumeric);
addParameter(parser, 'kappa3', 1000, @isnumeric);
addParameter(parser, 'proposal_sigma', 0.01, @isnumeric);
addParameter(parser, 'ar_lags', 4, @isnumeric);
parse(parser, varargin{:});
kappa1 = parser.Results.kappa1;
kappa2 = parser.Results.kappa2;
kappa3 = parser.Results.kappa3;
proposal_sigma = parser.Results.proposal_sigma;
ar_lags = parser.Results.ar_lags;

[Y,X,M,T,KK,~] = prepare_BVAR_matrices(Y,p,constant);

% Residual variances for Minnesota prior
s_squared = compute_residual_variances_MH(Y, ar_lags);

% SIGMA prior ~ IW(Psi_0, v_0)
Psi_0 = diag(s_squared);
v_0   = M + 2;

% Initialize at OLS
[A_OLS,SIGMA_OLS] = VAROLS(Y,X,M,p,constant);
A_current = A_OLS';
SIGMA = SIGMA_OLS;

A_samples     = zeros(num_samples, M, KK);
SIGMA_samples = zeros(num_samples, M, M);

fprintf('Running MH with %s loss and Minnesota prior (κ1=%.1f, κ2=%.1f)...\n', ...
        loss_fun, kappa1, kappa2);

q = 1e-7;   % regularization

for i = 1:num_samples
    if mod(i, 5000) == 0, fprintf('MCMC iteration %d/%d\n', i, num_samples); end

    % 1) RW-MH on A
    log_post_curr = log_posterior_minnesota(Y,X,A_current,q,M,p,constant,...
                                            kappa1,kappa2,kappa3,s_squared,loss_fun);
    for j = 1:10
        A_prop = A_current + proposal_sigma * randn(M,KK);
        log_post_prop = log_posterior_minnesota(Y,X,A_prop,q,M,p,constant, ...
                                                kappa1,kappa2,kappa3,s_squared,loss_fun);
        acc = exp(log_post_prop - log_post_curr);
        if rand < acc
            A_current = A_prop;
            log_post_curr = log_post_prop;
            break;
        end
    end

    % 2) Sample SIGMA via IW using loss-adjusted residual "sum"
    Resid = Y - X * A_current';
    switch lower(loss_fun)
        case 'ols'
            loss_A = Resid' * Resid;

        case 'lms'
            d = zeros(M,1);
            for eq = 1:M, d(eq) = median(Resid(:,eq).^2); end
            loss_A = diag(d);

        case 'lad'
            d = sum(abs(Resid), 1);
            loss_A = diag(d);

        case 'huber'
            delta = 1.345;
            H = zeros(1,M);
            for eq = 1:M
                r = Resid(:,eq); a = abs(r); is = (a <= delta);
                H(eq) = sum(0.5*r(is).^2) + sum(delta*(a(~is) - 0.5*delta));
            end
            loss_A = diag(H);

        case 'studentt'
            nu = 4;
            St = zeros(1,M);
            for eq = 1:M
                r = Resid(:,eq);
                St(eq) = -sum(log((1 + (r.^2)/nu).^(-(nu+1)/2)));
            end
            loss_A = diag(St);

        otherwise
            error('Unsupported loss function');
    end
    Psi_post = Psi_0 + loss_A;
    v_post   = v_0 + T;
    SIGMA    = iwishrnd(Psi_post, v_post);

    A_samples(i, :, :) = A_current;
    SIGMA_samples(i, :, :) = SIGMA;
end

A     = squeeze(mean(A_samples))';
SIGMA = squeeze(mean(SIGMA_samples));

%% ===== helpers (MH) =====
function s_squared = compute_residual_variances_MH(Y_local, ar_lags_local)
    [T_local, M_local] = size(Y_local);
    s_squared = zeros(M_local, 1);
    for i_local = 1:M_local
        y_local = Y_local(:, i_local);
        if ar_lags_local >= T_local
            s_squared(i_local) = var(y_local); continue;
        end
        Y_lag_local = zeros(T_local - ar_lags_local, ar_lags_local);
        for lag = 1:ar_lags_local
            Y_lag_local(:, lag) = y_local(ar_lags_local + 1 - lag : end - lag);
        end
        X_ar_local = [ones(T_local - ar_lags_local, 1), Y_lag_local];
        y_ar_local = y_local(ar_lags_local + 1 : end);
        beta_ar_local = (X_ar_local' * X_ar_local) \ (X_ar_local' * y_ar_local);
        resid_local = y_ar_local - X_ar_local * beta_ar_local;
        s_squared(i_local) = (resid_local' * resid_local) / max(1, (length(resid_local) - ar_lags_local - 1));
    end
end

function log_p = log_posterior_minnesota(Y_local,X_local,B_local,q_local,M_local,p_local,constant_local,kappa1_local,kappa2_local,kappa3_local,s_squared_local,lossfun_local)
    [n_local, m_local] = size(Y_local);  
    Resid_local = Y_local - X_local * B_local';

    % Loss part turned into a diagonal matrix
    switch lower(lossfun_local)
        case 'ols'
            loss_local = Resid_local' * Resid_local;

        case 'lms'
            d = zeros(m_local, 1);
            for eq = 1:m_local, d(eq) = median(Resid_local(:,eq).^2); end
            loss_local = diag(d);

        case 'lad'
            d = sum(abs(Resid_local), 1);
            loss_local = diag(d);

        case 'huber'
            delta = 1.345;
            H = zeros(1,m_local);
            for eq = 1:m_local
                r = Resid_local(:,eq); a = abs(r); is = (a <= delta);
                H(eq) = sum(0.5*r(is).^2) + sum(delta*(a(~is) - 0.5*delta));
            end
            loss_local = diag(H);

        case 'studentt'
            nu = 4;
            St = zeros(1,m_local);
            for eq = 1:m_local
                r = Resid_local(:,eq);
                St(eq) = -sum(log((1 + (r.^2)/nu).^(-(nu+1)/2)));
            end
            loss_local = diag(St);

        otherwise
            error('Unsupported loss function');
    end

    % Minnesota prior (diag precision per eq)
    log_prior = 0;
    for eq = 1:m_local
        V_inv = build_minnesota_precision_MH(eq, m_local, p_local, constant_local, ...
                                             kappa1_local, kappa2_local, kappa3_local, s_squared_local);
        b = B_local(eq,:)';
        log_prior = log_prior - 0.5 * (b' * V_inv * b) - 0.5 * log(det(2*pi * inv(V_inv)+1e-12*eye(size(V_inv))));
    end

    log_like = - (n_local / 2) * log(det(q_local * eye(m_local) + loss_local));
    log_p = log_prior + log_like;
end

function V_inv = build_minnesota_precision_MH(eq_idx, M_local, p_local, constant_local, kappa1_local, kappa2_local, kappa3_local, s_squared_local)
    KK_local = constant_local + M_local * p_local;
    V_inv = zeros(KK_local, KK_local);
    idx = 1;
    if constant_local
        V_inv(idx, idx) = 1 / kappa3_local; idx = idx + 1;
    end
    for lag = 1:p_local
        for var = 1:M_local
            if var == eq_idx
                prior_var = kappa1_local / (lag^2 * s_squared_local(eq_idx));
            else
                prior_var = kappa2_local / (lag^2 * s_squared_local(var));
            end
            V_inv(idx, idx) = 1 / prior_var;
            idx = idx + 1;
        end
    end
end
end
