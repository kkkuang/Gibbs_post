function eta = calibrate_eta_simple(Y, X, loss_fun)
% CALIBRATE_ETA_SIMPLE  Fast, deterministic η̂ for generalized Bayes.
%  Heuristic: η = trace(I)/trace(J), using loss score ψ and ψ' at OLS residuals.
%  Clipped to [0.1, 10] by default.

% Stable OLS and vectorization without big repmat
A_ols = (X.'*X + 1e-6*eye(size(X,2))) \ (X.'*Y);   % backslash over inv
E     = Y - X*A_ols;                                % T×M
rowSq = sum(X.^2,2);                                % T×1
rowSq = repmat(rowSq, size(Y,2), 1);                % TM×1 (small alloc)

% Guard J against negativity by taking expected positive curvature proxy
switch lower(loss_fun)
  case 'studentt'
    nu = 4;
    u  = E(:);
    psi  = (nu+1).*u./(nu+u.^2);
    psip = (nu+1).*(nu-u.^2)./(nu+u.^2).^2;
    psip = max(psip, 0);   % avoid negative curvature dominating J
  % ... other cases unchanged ...
end

Itrace = sum(psi.^2 .* rowSq);
Jtrace = max(sum(psip .* rowSq), 1e-12);
eta    = max(min(Itrace / Jtrace, 10), 0.1);
end
