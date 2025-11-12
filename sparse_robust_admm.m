function betaR = sparse_robust_admm(ts,p,constant,tau,lambda,epsilon,max_itr)
% SPARSE_ROBUST_ADMM  Robust+Sparse VAR via truncated data + ADMM grid
%
% betaR = sparse_robust_admm(ts,p,constant,tau,lambda,epsilon,max_itr)
%  • ts        : T×n data
%  • p         : lags
%  • constant  : 1/0 include intercept
%  • tau       : scalar, 1×n vector, or R×n matrix of truncation levels
%  • lambda    : vector of ℓ1 penalties (length L >= 1)
%  • epsilon   : ADMM tolerance (default 1e-5)
%  • max_itr   : ADMM max iterations (default 1000)
%
% Notes:
%  - This is a grid search over (tau,lambda). We guard against empty/NaN
%    grids and normalize shapes so nothing silently drops to length 0.
%  - Uses sparse_admm_seq.m internally on the (Σ0,Σ1) sufficient statistics.

    if nargin < 6 || isempty(epsilon), epsilon = 1e-5; end
    if nargin < 7 || isempty(max_itr), max_itr = 1000; end

    % ---------- grid guards ----------
    if isempty(lambda) || ~isvector(lambda) || any(~isfinite(lambda))
        error('sparse_robust_admm:lambda','Provide a non-empty finite vector "lambda".');
    end
    lambda = lambda(:).';                       % row vector
    L = numel(lambda);

    [T,n] = size(ts);
    if isscalar(tau)
        tau_mat = repmat(tau,1,n);
    elseif isvector(tau)
        tau_mat = reshape(tau,1,[]);
        if numel(tau_mat) ~= n
            error('sparse_robust_admm:tau','If tau is a vector it must be length n=%d.',n);
        end
    else
        if size(tau,2) ~= n
            error('sparse_robust_admm:tau','tau must have n=%d columns.',n);
        end
        tau_mat = tau;
    end
    R = size(tau_mat,1);
    if R==0
        error('sparse_robust_admm:tau','Empty tau grid.');
    end

    % ---------- design ----------
    [Y,X,~,~,~,~] = prepare_BVAR_matrices(ts,p,constant);
    Teff = size(Y,1);

    % ---------- precompute Σ0,Σ1 per truncation ----------
    output_A = cell(R*L,1);
    for r = 1:R
        % truncate responses (row-wise tau)
        y_tr = sign(Y).*min(abs(Y), tau_mat(r,:));
        % sufficient statistics
        Sigma0_tr = (X.'*X) / (Teff);
        Sigma1_tr = (X.'*y_tr) / (Teff);
        % run the ADMM sequence over lambda
        block = sparse_admm_seq(Sigma0_tr.', Sigma1_tr.', p, n, constant, lambda, epsilon, max_itr);
        output_A( (r-1)*L + (1:L) ) = block(:);
    end

    % ---------- rolling validation ----------
    T_val = max(10, round(0.5*Teff));   % last 50% to evaluate
    MSFE  = inf(numel(output_A),1);
    for i = 1:numel(output_A)
        Ai = output_A{i}.';             % KK×n
        sfe = 0;
        for t = Teff-T_val+1 : Teff
            e   = Y(t,:) - X(t,:)*Ai;
            sfe = sfe + mean(e.^2);
        end
        MSFE(i) = sfe / T_val;
    end

    % ---------- pick winner ----------
    [~,ind] = min(MSFE);
    betaR = output_A{ind}.';            % KK×n
end
