function output_A = sparse_admm_seq(Sigma0_tr, Sigma1_tr, p, d, constant, lambda_seq, epsilon, max_itr)
 % Goal: solve a sequence over lambda of
%   min_A  ||A*Sigma0_tr - Sigma1_tr||_1  + lambda * ||A||_1
% using a split with residual D:  A*Sigma0_tr - Sigma1_tr = D
% ADMM (ρ is penalty; λ is sparsity on A). We use soft-thresholding on A and D.

if nargin < 7 || isempty(epsilon), epsilon = 1e-5; end
if nargin < 8 || isempty(max_itr), max_itr = 1000; end

A_init = Sigma1_tr / Sigma0_tr;   % stable init if Sigma0_tr well-conditioned
A = A_init;
W = zeros(size(Sigma1_tr));       % dual for constraint A*S0 - S1 - D = 0
D = zeros(size(Sigma1_tr));

rho = 1.0;                        % NEW: ADMM penalty parameter (separate from lambda)

S0 = Sigma0_tr;
S0tS0 = S0 * S0.';                % for Lipschitz (if S0 is symmetric this is S0^2)
Lconst = rho * max(eig((S0.')*S0));   % Lipschitz for gradient in A-step
Lconst = max(Lconst, eps);

soft = @(x,t) sign(x) .* max(abs(x)-t, 0);

output_A = cell(length(lambda_seq), 1);

for lambda_ind = 1:length(lambda_seq)
    lam = lambda_seq(lambda_ind);
    A = A_init; D(:) = 0; W(:) = 0;

    for it = 1:max_itr
        % === A-step (prox-grad on 0.5*ρ||A*S0 - (S1 + D - W/ρ)||_F^2 + λ||A||_1) ===
        Rtarget = Sigma1_tr + D - W / rho;               % target for residual
        GradA   = rho * (A*S0 - Rtarget) * S0.';         % gradient wrt A
        A_new   = soft( A - (1/Lconst) * GradA, lam / Lconst );

        % === D-step (soft on residual LAD) ===
        R = A_new*S0 - Sigma1_tr + W / rho;              % current residual
        D_new = soft(R, 1/rho);

        % === Dual update ===
        W = W + rho * (A_new*S0 - Sigma1_tr - D_new);

        % === Convergence checks (relative) ===
        r = A_new*S0 - Sigma1_tr - D_new;                % primal residual
        s = rho * (D_new - D) * S0.';                    % dual-like change (proxy)
        nrm = max(1, norm(Sigma1_tr,'fro'));
        if max(norm(r,'fro')/nrm, norm(s,'fro')/max(1,norm(W,'fro'))) < epsilon ...
                && norm(A_new - A,'fro')/max(1,norm(A,'fro')) < 10*epsilon
            A = A_new; D = D_new; 
            break;
        end
        A = A_new; D = D_new;
    end

    output_A{lambda_ind} = A;
end
end
