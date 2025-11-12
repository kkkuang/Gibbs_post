function out = apply_outliers_and_estimate_VAR(Y_clean, opts)
% APPLY_OUTLIERS_AND_ESTIMATE_VAR  (GRID VERSION, filtered NV pairs)
% Runs a set of estimators (method × loss) on the same Y, after optional contamination.
%
% Required fields in opts:
%   .p, .cons
%   .estimators : array of structs with fields {id, loss, label}
% Contamination (optional):
%   .outlier_type, .contamination_rate, .outlier_magnitude
 
% --------------------------- defaults -------------------------------------
args = struct( ...
    'p', 4, 'cons', 1, ...
    'outlier_type', 1, 'contamination_rate', 0.05, ...
    'outlier_magnitude', 3, 'delta', 0.7, ...
    'estimator_names', {{'OLS'}}, ...
    'scale_mode', 'resid' ...
);
if nargin >= 2 && ~isempty(opts)
    f = fieldnames(opts);
    for i = 1:numel(f), args.(f{i}) = opts.(f{i}); end
end
p      = args.p;  cons = args.cons;
otype  = args.outlier_type; cr = args.contamination_rate;
mag    = args.outlier_magnitude; delta = args.delta;
enames = args.estimator_names;   E = numel(enames);

[T, n] = size(Y_clean);

% ----------------- baseline VAR on clean data (beta_true) -----------------
[y_clean, X_clean, ~, ~, ~, ~] = prepare_BVAR_matrices(Y_clean, p, cons);
[beta_true, sigma_true] = VAROLS(y_clean, X_clean, n, p, cons);
K = size(beta_true,1);

% Scale used for outliers
switch lower(args.scale_mode)
    case 'resid'
        resid = y_clean - X_clean*beta_true; data_std = std(resid, 0, 1);
    case 'diff'
        data_std = std(diff(Y_clean), 0, 1);
    otherwise
        data_std = std(Y_clean, 0, 1);
end
data_std(data_std==0) = 1;

% -------------------------- contaminate once ------------------------------
Y_cont = Y_clean;
n_out = max(0, round(cr * T));
outlier_locs = []; w_used = [];

if n_out > 0
    pool = (p+1):T; n_out = min(n_out, numel(pool));
    outlier_locs = sort(randsample(pool, n_out));
    W = (2*rand(n_out, n) - 1) .* (mag .* data_std);
    switch otype
        case 1 % MAO
            for i=1:n_out, Y_cont(outlier_locs(i),:) = Y_cont(outlier_locs(i),:) + W(i,:); end
        case 2 % MIO (innovational, MA(∞) propagation)
            if cons, Phi = beta_true(2:end,:); else, Phi = beta_true; end
            A_list = zeros(n,n,p);
            for L=1:p, A_list(:,:,L) = Phi((L-1)*n+(1:n), :)'; end
            Psi = local_VAR_MA_coefs(A_list, T);
            for i=1:n_out
                h = outlier_locs(i); w = W(i,:).';
                for j=0:(T-h), Y_cont(h+j,:) = Y_cont(h+j,:) + (Psi{j+1}*w).'; end
            end
        case 3 % MLS
            for i=1:n_out, h=outlier_locs(i); Y_cont(h:end,:) = Y_cont(h:end,:) + W(i,:); end
        case 4 % MTC
            for i=1:n_out
                h=outlier_locs(i);
                for t=h:T, Y_cont(t,:) = Y_cont(t,:) + (delta^(t-h)) * W(i,:); end
            end
        otherwise, error('Unknown outlier_type %d', otype);
    end
    w_used = W;
end

% -------------------------- re-estimate on Y_cont -------------------------
[ y_est, X_est, ~, ~, ~, ~ ] = prepare_BVAR_matrices(Y_cont, p, cons);

beta_est = NaN(K, n, E);
mse      = NaN(1, E);

for e = 1:E
    nm = enames{e};
    id = lower(nm);  % crude parser for loss tag
    % map name -> loss string
    if contains(id,'huber'), L='huber';
    elseif contains(id,'lad'), L='lad';
    elseif contains(id,'studentt') || contains(id,'student-t'), L='studentt';
    else, L='ols';
    end

    % compute η for this loss once
    eta_e = calibrate_eta_simple(y_est, X_est, L);

    try
        switch lower(nm)
            case 'ols'
                [be, ~] = VAROLS(y_est, X_est, n, p, cons);

            case 'huber'
                [be, ~, ~] = robust_var(Y_cont, p, cons, 'loss','huber','cov_method','mcd','verbose', false);
                % (point estimator; η not used)

            case 'robust sparse'
                tau_seq = quantile(abs(Y_cont), [0.5, 0.75, 0.9, 0.95, 1]);
                tau_com = combos(tau_seq);
                lambda_seq = logspace(-2, 3, 10);
                be = sparse_robust_admm(Y_cont, p, cons, tau_com, lambda_seq, 1e-5, 1000);

            case 'lasso'
                lambda_seq = logspace(-2, 3, 10);
                be = Lasso_VAR(Y_cont, p, cons, lambda_seq, 1e-20, 1000);

            case 'bvar-ald'
                % Temper ALD by η (add 'eta', implemented in your BVARquantile)
                [be, ~, ~, ~] = BVARquantile(Y_cont, p, cons, 0.5*ones(1,n), 20000, ...
                                             'prior_type','noninformative', 'eta', eta_e);

            case 'bvar-ald-l(1/2)'
                [be, ~, ~, ~] = BVARquantile(Y_cont, p, cons, 0.5*ones(1,n), 20000, ...
                                             'eta', eta_e);  % if your code supports L(1/2), temper likewise

            case 'bvar-huber'
                [be, ~] = BVARlossLLB(Y_cont, p, cons, 'huber', 20000, ...
                                      'calibrate_w', false, 'eta', eta_e);

            case 'bvar-minn-huber'
                [be, ~] = BVARlossLLB(Y_cont, p, cons, 'huber', 20000, ...
                                      'calibrate_w', false, 'eta', eta_e, ...
                                      'kappa1',0.1,'kappa2',0.1);

            case 'bvar-ssvs-huber'
                % LLB-SSVS version should accept 'eta' and pass into LLB core
                [be, ~] = BVARlossLLB_ssvs(Y_cont, p, cons, 'huber', 20000, ...
                                           'eta', eta_e);

            case 'bvar-smc-huber'
                [be, ~] = BVARlossSMC(Y_cont, p, cons, 'huber', 10000, 'eta', eta_e);

            case 'bvar-vb-huber'
                [be, ~] = BVARlossVB(Y_cont, p, cons, 'huber', 'variational_family','mean_field', ...
                                      'max_iter', 5000, 'eta', eta_e);

            case 'bvar-mh'
                [be, ~] = BVARlossMH(Y_cont, p, cons, L, 20000, 'eta', eta_e);

            otherwise
                error('Estimator "%s" not implemented here.', nm);
        end

        beta_est(:,:,e) = be;
        mse(e) = mean((be(:) - beta_true(:)).^2);
    catch ME
        warning('Estimator %s failed: %s', nm, ME.message);
    end
end

% -------------------------- package output --------------------------------
out = struct();
out.beta_true = beta_true;
out.sigma_true = sigma_true;
out.beta_est = beta_est;
out.mse = mse;
out.Y_cont = Y_cont;
out.outlier_locs = outlier_locs;
out.w_used = w_used;
out.estimator_names = enames;
out.meta = struct('p',p,'cons',cons,'outlier_type',otype,'contamination_rate',cr, ...
                  'outlier_magnitude',mag,'delta',delta,'scale_mode',args.scale_mode);
end

% ---- local MA(∞) helper ----
function Psi = local_VAR_MA_coefs(A_list, J)
n = size(A_list,1); p = size(A_list,3);
Psi = cell(J+1,1); Psi{1} = eye(n);
for j=1:J
    S = zeros(n); for i=1:min(j,p), S = S + Psi{j-i+1} * A_list(:,:,i); end
    Psi{j+1} = S;
end
end
