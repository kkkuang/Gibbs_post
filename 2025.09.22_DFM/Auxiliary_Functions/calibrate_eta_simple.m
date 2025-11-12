function eta = calibrate_eta_simple(Y, X, loss_fun)
% CALIBRATE_ETA_SIMPLE  Fast, deterministic η̂ for generalized Bayes.
%  Heuristic: η = trace(I)/trace(J), using loss score ψ and ψ' at OLS residuals.
%  Clipped to [0.1, 10] by default.

A_ols = (X'*X + 1e-6*eye(size(X,2))) \ (X'*Y);
E = Y - X*A_ols;              % T×M
u = E(:);                     % TM×1

switch lower(loss_fun)
  case 'ols'
    psi = u;                            psip = ones(size(u));
  case 'lad'
    psi = sign(u);                      psip = zeros(size(u));
  case 'huber'
    d = 1.345; a = abs(u);
    psi = min(max(u, -d), d);           psip = double(a <= d);
  case 'studentt'
    nu=4; psi = (nu+1).*u./(nu+u.^2);   psip = (nu+1).*(nu-u.^2)./(nu+u.^2).^2;
  otherwise
    psi = u; psip = ones(size(u));
end

% Fast traces (stacked design trick)
Xrep = repmat(X, size(Y,2), 1);
rowSq = sum(Xrep.^2, 2);
Itrace = sum(psi.^2 .* rowSq);
Jtrace = sum(psip    .* rowSq);

eta = Itrace / max(Jtrace, 1e-12);
eta = max(min(eta, 10), 0.1);
end
