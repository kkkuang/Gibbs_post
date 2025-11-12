function [A,SIGMA] = BVARlossMH_ssvs(Y,p,constant,loss_fun,num_samples,varargin)
% BVARlossMH_ssvs  — SSVS VAR via RW-MH with robust loss (OLS/LMS/LAD/HUBER)
%
% New: supports 'huber' with tuning c (default 1.345) and per-equation MAD scale.

% ----------------------- options -----------------------
pr = inputParser;
addParameter(pr,'proposal_sigma',0.01,@isnumeric);
addParameter(pr,'q_reg',1e-7,@isnumeric);          % small ridge in log|qI + loss|
addParameter(pr,'tau0',sqrt(0.1),@isnumeric);      % SSVS spike sd
addParameter(pr,'tau1',sqrt(4),@isnumeric);        % SSVS slab sd
addParameter(pr,'pi0',0.5,@isnumeric);             % prior inclusion prob
addParameter(pr,'huber_c',1.345,@isnumeric);       % Huber tuning constant
parse(pr,varargin{:});
prop_sig = pr.Results.proposal_sigma;
q        = pr.Results.q_reg;
tau0     = pr.Results.tau0;
tau1     = pr.Results.tau1;
pi0      = pr.Results.pi0;
cHuber   = pr.Results.huber_c;

% ----------------------- data prep ----------------------
[Y,X,M,T,KK,~] = prepare_BVAR_matrices(Y,p,constant);

% priors
Psi_0 = eye(M);
v_0   = M + 2;

% init at OLS
[A_OLS,SIGMA_OLS] = VAROLS(Y,X,M,p,constant);
A_current = A_OLS';             % M x KK
SIGMA     = SIGMA_OLS;
GAMMA     = ones(size(A_current));

A_samples     = zeros(num_samples,M,KK);
SIGMA_samples = zeros(num_samples,M,M);

% ----------------------- MCMC ---------------------------
for i = 1:num_samples
    % --- 1) RW–MH on A (pseudo-likelihood under chosen loss) ---
    s_vec = robust_scales(Y - X*A_current');  % 1xM MAD scales for robust losses
    lp_curr = log_post_ssvs(Y,X,A_current,GAMMA,q,tau0,tau1,loss_fun,cHuber,s_vec);

    for j = 1:10
        A_prop = A_current + prop_sig * randn(M,KK);
        lp_prop = log_post_ssvs(Y,X,A_prop,GAMMA,q,tau0,tau1,loss_fun,cHuber,s_vec);
        if log(rand) < (lp_prop - lp_curr)
            A_current = A_prop; lp_curr = lp_prop; break;
        end
    end

    % --- 2) sample SSVS indicators Γ ---
    for ieq = 1:M
        l0  = lnormpdf(A_current(ieq,:),0,tau0);
        l1  = lnormpdf(A_current(ieq,:),0,tau1);
        pip = 1 ./ (1 + ((1-pi0)./pi0).*exp(l0 - l1));
        GAMMA(ieq,:) = binornd(1,pip);
    end

    % --- 3) update SIGMA (IW with loss-based diagonal surrogate) ---
    Resid = Y - X*A_current';
    switch lower(loss_fun)
        case 'ols'
            loss_A = Resid' * Resid;                           % MxM
        case 'lms'
            d = median(reshape(Resid.^2,T,M)); loss_A = diag(d);
        case 'lad'
            d = sum(abs(Resid),1);               loss_A = diag(d);
        case 'huber'
            % weighted sum of squares per eq using Huber weights
            s = max(robust_scales(Resid), 1e-6);               % 1xM
            W = min(1, (cHuber ./ max(abs(Resid)./s, 1e-12))); % T x M
            d = sum(W .* (Resid.^2), 1);        loss_A = diag(d);
        otherwise
            error('Unsupported loss function: %s',loss_fun);
    end
    Psi_post = Psi_0 + loss_A;
    v_post   = v_0 + T;
    SIGMA    = iwishrnd(Psi_post, v_post);

    % store
    A_samples(i,:,:)   = A_current;
    SIGMA_samples(i,:,:)= SIGMA;
end

A     = squeeze(mean(A_samples,1))';
SIGMA = squeeze(mean(SIGMA_samples,1));

end

% ----------------------- helpers ------------------------
function lp = log_post_ssvs(Y,X,B,Gamma,q,tau0,tau1,lossfun,cHuber,s_vec)
    [n,m] = size(Y);
    R = Y - X*B';                   % n x m
    switch lower(lossfun)
        case 'ols'
            L = R' * R;             % m x m
        case 'lms'
            L = diag(median(reshape(R.^2,n,m)));
        case 'lad'
            L = diag(sum(abs(R),1));
        case 'huber'
            s = max(s_vec, 1e-6);   % 1 x m
            r = R ./ s;             % standardized residuals
            % Huber rho: 0.5 r^2 (|r|<=c), else c(|r|-0.5c)
            a = abs(r);
            rho = 0.5*(a<=cHuber).*(r.^2) + (a>cHuber).*(cHuber.*a - 0.5*cHuber^2);
            % Convert to a diagonal loss surrogate on original scale
            d = sum(2*rho .* (s.^2), 1);    % ≈ weighted sum of squares
            L = diag(d);
        otherwise
            error('Unsupported loss function: %s',lossfun);
    end

    % SSVS prior for rows of B
    log_prior = 0;
    for j = 1:m
        Dj = Gamma(j,:).*tau1 + (1-Gamma(j,:)).*tau0;
        log_prior = log_prior + sum(lnormpdf(B(j,:),0,Dj));
    end

    lp = log_prior - (n/2) * log(det(q*eye(m) + L));
end

function s = robust_scales(E)
    % Per-column MAD scale with Gaussian consistency 1.4826
    s = 1.4826 * mad(E,1,1);   % 1 x M, (median abs dev about median)
    s(~isfinite(s) | s<=0) = std(E,0,1,'omitnan');
    s(~isfinite(s) | s<=0) = 1;
end
