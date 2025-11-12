function [beta, sigma, diagnostics] = robust_var(Y, p, const, varargin)
% ROBUST_VAR  System-wide robust VAR(p) via IRLS + robust scatter
%
% [beta, sigma, diagnostics] = robust_var(Y, p, const, 'loss',LOSS, 'cov_method',COV, ...)
% LOSS: 'ols'|'lad'|'huber'|'bisquare'|'cauchy'|'fair'|'studentt'
% COV : 'mcd' (fmcd) | 'ogk' | 'olivehawkins' | 'classical'
%
% Name-value (selected):
%   'tuning'  - scalar tuning (Huber c, Bisquare c, Cauchy c, Fair c). Ignored for 'ols' and 'lad'.
%   'nu'      - degrees of freedom for 'studentt' (default 4)
%   'maxiter' - IRLS max iters (default 100)
%   'tol'     - IRLS tol on objective change (default 1e-6)
%   'verbose' - print progress/warnings (default false)

parser = inputParser;
addRequired(parser, 'Y', @isnumeric);
addRequired(parser, 'p', @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addRequired(parser, 'const', @(x)isnumeric(x)&&(x==0||x==1));
addParameter(parser, 'loss', 'huber', @(s)ischar(s)||isstring(s));
addParameter(parser, 'cov_method', 'mcd', @(s)ischar(s)||isstring(s));
addParameter(parser, 'tuning', [], @isnumeric);
addParameter(parser, 'nu', 4, @isnumeric);
addParameter(parser, 'maxiter', 100, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(parser, 'tol', 1e-6, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(parser, 'verbose', false, @islogical);
parse(parser, Y, p, const, varargin{:});
opts = parser.Results;

% Design matrices
[y, X] = prepare_var_mats(Y, p, const);
[T_est, k] = size(X);
n = size(Y,2);

% Diagnostics container
diagnostics = struct('loss_function',lower(string(opts.loss)), ...
                     'cov_method',lower(string(opts.cov_method)), ...
                     'converged',false,'iterations',0, ...
                     'objective_values',[],'tuning_parameter',[]);

% Default tuning per loss
tuning = default_tuning(opts.loss, opts.tuning);
diagnostics.tuning_parameter = tuning;

% Initial OLS
beta = (X' * X) \ (X' * y);
resid = y - X * beta;

if strcmpi(opts.loss,'ols')
    sigma = cov_weighted(resid, []); % classical
    sigma = regularize_sigma(sigma);
    diagnostics.converged = true;
    diagnostics.iterations = 1;
    diagnostics.weights = ones(T_est,1);
    return
end

% Initial scatter (robust if feasible)
sigma = robust_scatter(resid, opts.cov_method, opts.verbose);
sigma = regularize_sigma(sigma);

% IRLS
obj_old = Inf;
w = ones(T_est,1);

for it = 1:opts.maxiter
    % 1) weights from multivariate residuals
    w = obs_weights(resid, sigma, opts.loss, tuning, opts.nu);

    % 2) WLS update equation-by-equation (stable scaling)
    beta = wls_update(y, X, w);

    % 3) residuals and scatter update
    resid = y - X * beta;
    sigma = robust_scatter(resid, opts.cov_method, opts.verbose);
    sigma = regularize_sigma(sigma);

    % 4) objective
    obj = objective_val(resid, sigma, opts.loss, tuning, opts.nu);
    diagnostics.objective_values(end+1,1) = obj; %#ok<AGROW>

    if opts.verbose && mod(it,10)==0
        fprintf('[robust_var] iter %d: obj=%.6f  Δ=%.2e\n', it, obj, abs(obj-obj_old));
    end
    if abs(obj - obj_old) < max(opts.tol, 1e-12)*(1 + abs(obj))
        diagnostics.converged = true; break;
    end
    obj_old = obj;
end
diagnostics.iterations = it;
diagnostics.final_objective = obj;
diagnostics.weights = w;
diagnostics.residuals = resid;

% ---- helpers -------------------------------------------------------------
function [y, X] = prepare_var_mats(Y, p, cflag)
    [T,n] = size(Y);
    y = Y(p+1:end,:);
    X = [];
    if cflag, X = [X, ones(T-p,1)]; end
    for L=1:p
        X = [X, Y(p+1-L:T-L,:)]; %#ok<AGROW>
    end
end

function t = default_tuning(loss, user_t)
    if ~isempty(user_t), t = user_t; return; end
    switch lower(loss)
        case 'lad',      t = 1;
        case 'huber',    t = 1.345;
        case 'bisquare', t = 4.685;
        case 'cauchy',   t = 2.385;
        case 'fair',     t = 1.4;
        case 'studentt', t = NaN; % uses opts.nu
        otherwise,       t = 1;
    end
end

function w = obs_weights(R, S, loss, t, nu)
    % Mahalanobis distance per row via chol inverse (more stable than pinv)
    [T,~] = size(R);
    w = ones(T,1);
    S = regularize_sigma(S);
    % compute S^{-1/2}
    try
        L = chol(S,'lower');
        Linv = L \ eye(size(S));
        Z = R * Linv';           % whitened residuals
        d = sqrt(sum(Z.^2,2));
    catch
        % fallback to pseudo-inverse if chol fails
        Sinv = pinv(S);
        d = sqrt(sum((R*Sinv).*R,2));
    end

    switch lower(loss)
        case 'lad'
            w = 1 ./ max(d,1e-6);
        case 'huber'
            w = min(1, t ./ max(d,1e-6));
        case 'bisquare'
            u = d/t; w = (d<=t) .* (1 - u.^2).^2;
        case 'cauchy'
            w = 1 ./ (1 + (d/t).^2);
        case 'fair'
            w = 1 ./ (1 + d/max(t,1e-6));
        case 'studentt'
            if isempty(nu) || ~isfinite(nu), nu = 4; end
            w = (nu + size(R,2)) ./ (nu + d.^2);
        otherwise
            error('Unknown loss function: %s', loss);
    end
    % clamp extremely small/large weights
    w = max(w, 1e-6);
end

function B = wls_update(y, X, w)
    W  = sqrt(w(:));
    Xw = X .* W;                 % row-scale
    Yw = (y .* W);
    B  = Xw \ Yw;                % QR under the hood
end

function S = robust_scatter(R, method, verbose)
    % Stable chain: FMCD -> OGK -> OliveHawkins -> classical
    [T,p] = size(R);
    meth = lower(method);
    h_default = ceil((T + p + 1)/2);
    fmcd_ok = (h_default > p) && T >= p+2;

    try
        switch meth
            case {'mcd','fmcd'}
                if ~fmcd_ok, error('fmcd_not_feasible'); end
                alpha = 0.25; % 25% contamination → h ≈ 0.75·T
                [S,~] = robustcov(R, 'Method','fmcd', ...
                                     'OutlierFraction', alpha, ...
                                     'NumTrials', 500, ...
                                     'BiasCorrection', 1);
            case 'ogk'
                [S,~] = robustcov(R, 'Method','ogk', 'NumOGKIterations', 2, ...
                                        'UnivariateEstimator', 'tauscale');
            case 'olivehawkins'
                [S,~] = robustcov(R, 'Method','olivehawkins', ...
                                     'NumConcentrationSteps', 10, ...
                                     'ReweightingMethod','rfch');
            otherwise
                S = cov_weighted(R, []);
        end
    catch
        if ~strcmpi(meth,'ogk')
            try
                [S,~] = robustcov(R, 'Method','ogk', 'NumOGKIterations', 1);
                if verbose, fprintf('[robust_var] robustcov(%s) failed; used OGK.\n', meth); end
            catch
                try
                    [S,~] = robustcov(R, 'Method','olivehawkins');
                    if verbose, fprintf('[robust_var] robustcov(%s) failed; used Olive-Hawkins.\n', meth); end
                catch
                    S = cov_weighted(R, []);
                    if verbose, fprintf('[robust_var] robustcov(%s) failed; used classical covariance.\n', meth); end
                end
            end
        else
            try
                [S,~] = robustcov(R, 'Method','olivehawkins');
                if verbose, fprintf('[robust_var] robustcov(OGK) failed; used Olive-Hawkins.\n'); end
            catch
                S = cov_weighted(R, []);
                if verbose, fprintf('[robust_var] robustcov(OGK) failed; used classical covariance.\n'); end
            end
        end
    end
end

function S = cov_weighted(R, w)
    % classical (optionally weighted) covariance
    [T,p] = size(R);
    if isempty(w)
        R0 = R - mean(R,1);
        S  = (R0' * R0) / max(T-1,1);
    else
        w = w(:);
        w = max(w,0);                             % guard
        sw = sum(w);
        if sw<=0, w = ones(T,1)/T; else, w = w/sw; end
        mu = sum(R .* w, 1);
        R0 = R - mu;
        denom = 1 - sum(w.^2) + eps;
        S  = (R0' * (R0 .* w)) / denom;
    end
end

function S = regularize_sigma(S)
    S = (S + S')/2;
    [V,D] = eig(S);
    d = max(diag(D), 1e-8);
    S = V * diag(d) * V';
    S = (S + S')/2;
end

function obj = objective_val(R, S, loss, t, nu)
    S = regularize_sigma(S);
    % whitened residuals for stable Mahalanobis
    try
        L = chol(S,'lower'); Linv = L \ eye(size(S)); Z = R * Linv';
        d = sqrt(sum(Z.^2,2));
    catch
        Sinv = pinv(S); d = sqrt(sum((R*Sinv).*R,2));
    end
    switch lower(loss)
        case 'ols'
            rho = 0.5 * d.^2;
        case 'lad'
            rho = abs(d);
        case 'huber'
            a = abs(d);
            rho = (a<=t).* (0.5*d.^2) + (a>t).*(t*a - 0.5*t^2);
        case 'bisquare'
            u = d/t;
            rho = (abs(d)<=t) .* ((t^2/6) * (1 - (1 - u.^2).^3)) + (abs(d)>t) .* (t^2/6);
        case 'cauchy'
            rho = (t^2/2) * log(1 + (d/t).^2);
        case 'fair'
            rho = t^2 * (abs(d)/t - log(1 + abs(d)/t));
        case 'studentt'
            if isempty(nu) || ~isfinite(nu), nu = 4; end
            rho = 0.5*(nu + n) * log(1 + (d.^2)/nu);
        otherwise
            error('Unknown loss function: %s', loss);
    end
    obj = sum(rho);
end

end
