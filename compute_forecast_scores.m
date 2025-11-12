function [scores, loss_seq] = compute_forecast_scores(Y, beta_est_all, p, cons)
% COMPUTE_FORECAST_SCORES  (DM-friendly)
% Output:
%   scores.mse1/mae1/crps1/lps1  : 1 x E
%   scores.mse4/mae4, mse8/mae8  : 1 x E
%   loss_seq.mse1{e}, .mae1{e}   : per-time vectors (for DM tests)

[T, n] = size(Y);
E = size(beta_est_all, 3);

% Build full matrices once
[y_all, X_all] = prepare_BVAR_matrices(Y, p, cons);  % (T-p) x n, (T-p) x K
Teff = size(y_all,1);
if Teff < 30
    scores = empty_scores(E); loss_seq = empty_loss_seq(E);
    return;
end

% Holdout split
Ttest  = min(max(10, floor(Teff/3)), 60);
Ttrain = Teff - Ttest;
y_tr = y_all(1:Ttrain, :);
X_tr = X_all(1:Ttrain, :);
y_te = y_all(Ttrain+1:end, :);
X_te = X_all(Ttrain+1:end, :);

% One common plug-in covariance from OLS on training
[~, Sigma_plugin] = VAROLS(y_tr, X_tr, n, p, cons);
Sigma_plugin = (Sigma_plugin + Sigma_plugin')/2 + 1e-8*eye(n);

% Prealloc
mse1  = NaN(1,E); mae1  = NaN(1,E); crps1 = NaN(1,E); lps1 = NaN(1,E);
mse4  = NaN(1,E); mae4  = NaN(1,E);
mse8  = NaN(1,E); mae8  = NaN(1,E);

loss_seq.mse1 = cell(1,E);
loss_seq.mae1 = cell(1,E);

% 1-step predictive covariance pieces
[R, ~]   = chol(Sigma_plugin + 1e-8*eye(n));
logdetS  = 2*sum(log(diag(R)));
sdiag    = sqrt(max(diag(Sigma_plugin), 1e-12))';

for e = 1:E
    be = beta_est_all(:,:,e);
    if ~isfloat(be), be = double(be); end
    if isempty(be) || all(isnan(be(:)))
        loss_seq.mse1{e} = []; loss_seq.mae1{e} = [];
        continue;
    end

    % -------- 1-step (direct using X_te) --------
    yhat1 = X_te * be;                   % Ttest x n
    err1  = y_te - yhat1;                % Ttest x n
    loss_seq.mse1{e} = mean(err1.^2, 2); % per-t avg over series (DM input)
    loss_seq.mae1{e} = mean(abs(err1), 2);

    mse1(e) = mean(loss_seq.mse1{e});
    mae1(e) = mean(loss_seq.mae1{e});

    % CRPS (Gaussian, diagonal closed-form aggregated across series)
    cr = 0;
    for j = 1:n
        z = err1(:,j) ./ sdiag(j);
        cr = cr + mean(crps_gaussian(z, sdiag(j)));
    end
    crps1(e) = cr / n;

    % LPS (Gaussian, same plug-in)
    ll = 0;
    for t = 1:Ttest
        z = (err1(t,:) / R);                 % 1×n
        ll = ll - 0.5*(logdetS + z*z' + n*log(2*pi));
    end
    lps1(e) = ll / Ttest;

    % -------- multi-step iterated (h=4,8): MSE/MAE only --------
    if cons
        c = be(1,:);            % 1×n intercept
    else
        c = [];
    end
    [err4, err8] = iterated_var_errors(Y, be, c, p, cons, Ttrain, 4, 8);
    if ~isempty(err4)
        mse4(e) = mean(mean(err4.^2, 2));
        mae4(e) = mean(mean(abs(err4), 2));
    end
    if ~isempty(err8)
        mse8(e) = mean(mean(err8.^2, 2));
        mae8(e) = mean(mean(abs(err8), 2));
    end
end

scores = struct('mse1',mse1,'mae1',mae1,'crps1',crps1,'lps1',lps1, ...
                'mse4',mse4,'mae4',mae4,'mse8',mse8,'mae8',mae8);
end

% ---------- helpers ----------
function s = empty_scores(E)
s = struct('mse1',NaN(1,E),'mae1',NaN(1,E),'crps1',NaN(1,E),'lps1',NaN(1,E), ...
           'mse4',NaN(1,E),'mae4',NaN(1,E),'mse8',NaN(1,E),'mae8',NaN(1,E));
end
function l = empty_loss_seq(E)
l.mse1 = cell(1,E); l.mae1 = cell(1,E);
end

function v = crps_gaussian(z, sigma)
% z = (y-mu)/sigma, CRPS for N(mu, sigma^2)
v = sigma .* ( z .* (2*normcdf(z)-1) + 2*normpdf(z) - 1/sqrt(pi) );
end

function [err4, err8] = iterated_var_errors(Y, be, c, p, cons, Ttrain, h4, h8)
% Returns error matrices for horizons h4 and h8:
%   errH is (Ttest-H+1) x n
[~, n] = size(Y);
[y_all, ~] = prepare_BVAR_matrices(Y, p, cons);
Teff  = size(y_all,1);
Ttest = Teff - Ttrain;
if Ttest <= max(h4,h8)
    err4 = []; err8 = []; return;
end

% Companion coefficients A1..Ap from be
A_list = beta_to_A_list(be, n, p, cons);  % n x n x p

len4 = Ttest - h4 + 1;
len8 = Ttest - h8 + 1;
err4 = NaN(len4, n); err8 = NaN(len8, n);

for t_idx = 1:Ttest
    t0 = p + Ttrain + (t_idx-1);              % index in Y
    y_hist = flipud( Y(t0-p+1:t0, :) );       % p x n, most recent first
    ypath  = iterated_forecast_path(y_hist, A_list, c, cons, p, max(h4,h8));

    if t_idx <= len4
        err4(t_idx,:) = Y(t0 + h4, :) - ypath(h4, :);
    end
    if t_idx <= len8
        err8(t_idx,:) = Y(t0 + h8, :) - ypath(h8, :);
    end
end
end

function A_list = beta_to_A_list(be, n, p, cons)
% be: K x n, K = cons + n*p.
Phi = be(1 + cons:end, :);  % drop intercept row if present
A_list = zeros(n, n, p);
for lag = 1:p
    rows = (lag-1)*n + (1:n);
    A_list(:,:,lag) = Phi(rows, :)';
end
end

function ypath = iterated_forecast_path(y_hist, A_list, c, cons, p, H)
% y_hist: p x n (most recent first), A_list: n x n x p, c: 1×n intercept
n = size(A_list,1);
ypath = NaN(H, n);
y_lags = y_hist;

for h = 1:H
    yhat = zeros(1,n);
    for lag = 1:p
        yhat = yhat + (y_lags(lag,:) * A_list(:,:,lag)');
    end
    if cons
        yhat = yhat + c;   % add intercept each step
    end
    ypath(h,:) = yhat;

    % update lag stack (shift down, insert yhat on top)
    if p > 1
        y_lags(2:end,:) = y_lags(1:end-1,:);
    end
    y_lags(1,:) = yhat;
end
end
