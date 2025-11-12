function betaL = Lasso_VAR(ts,p,constant,lambda_seq,epsilon,max_itr) 
% (unchanged header)
if nargin < 5, epsilon = 1e-5; end
if nargin < 6, max_itr = 1000; end

[T,~] = size(ts);
[Y,X,~,~,~,~] = prepare_BVAR_matrices(ts,p,constant);

% Ridge-system precompute for ADMM:
% Solve min_A 0.5||Y - X A||_F^2 + lam ||A||_1 via ADMM with penalty rho
rho = 1.0;                              % NEW: ADMM penalty (was kappa)
Xtd = [X; sqrt(rho)*eye(size(X,2))];    % NEW: correct scaling
[Q,R] = qr(Xtd, 0);
Mmap  = (R \ Q');                       % inv(R)*Q' (stable)

A = X \ Y;           % warm start  (KK x n)
theta = A; c = zeros(size(A));

output_A = cell(length(lambda_seq), 1);

for i_lambda = 1:length(lambda_seq)
    lam = lambda_seq(i_lambda);
    A_old = A; theta = A; c(:) = 0;
    for i = 1:max_itr
        % A-update (ridge system)
        A = Mmap * [Y; sqrt(rho)*(theta - c)];

        % soft-threshold (vectorized) with lam/rho  (FIX)
        S = A + c;
        theta = sign(S) .* max(abs(S) - lam/rho, 0);

        % dual update
        c = c + A - theta;

        % stopping (relative)
        rel = max(norm(A - A_old,'fro'), norm(A - theta,'fro')) / max(1,norm(A,'fro'));
        if rel < epsilon, break; end
        A_old = A;
    end
    output_A{i_lambda} = A';   % store as n x KK
end

% simple rolling validation
Teff = size(Y,1);
T_val = max(10, round(0.2*Teff));    % last 20%
MSFE = inf(length(lambda_seq),1);

for i = 1:length(lambda_seq)
    Ai = output_A{i}';               % KK x n
    sfe = 0;
    for t = Teff - T_val + 1 : Teff
        e = Y(t,:) - X(t,:)*Ai;      % 1 x n
        sfe = sfe + mean(e.^2);
    end
    MSFE(i) = sfe / T_val;
end

[~,ind] = min(MSFE);
betaL = output_A{ind}';               % KK x n
end
