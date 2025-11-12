function W = iwishrnd(Psi, nu)
% Sample IW(Psi, nu) via Bartlett without explicit inverses
p   = size(Psi,1);
Psi = (Psi+Psi.')/2;
[U,pd] = chol(Psi,'lower');             % Psi = U*U.'
if pd>0, U = chol((Psi+1e-8*eye(p)),'lower'); end

% Bartlett for standard Wishart Wp(nu, I)
T = tril(randn(p));                      % fill lower; diagonals ~ sqrt(chi2)
for i=1:p, T(i,i) = sqrt(chi2rnd(nu - i + 1)); end
S = T*T.';                               % ~ Wishart(I, nu)

% Wishart with scale inv(Psi): S̃ = U^{-1} * S * U^{-T}
S_tilde = U \ (S / U.');                 % solves, no inv

% Inverse-Wishart draw is (S̃)^{-1}
[Rs,ps] = chol(S_tilde,'lower');
if ps==0
    W = Rs'\(Rs\eye(p));                 % inv via two triangular solves
else
    W = (S_tilde + 1e-8*eye(p)) \ eye(p);
end
W = (W+W.')/2;
end
