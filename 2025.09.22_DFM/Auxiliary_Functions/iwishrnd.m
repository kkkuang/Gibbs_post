function W = iwishrnd(Psi, nu)
% Robust IW draw with ridge & symmetry guards
p = size(Psi,1);
Psi = (Psi+Psi')/2; 
[V,D] = eig(Psi); d = max(real(diag(D)), 1e-8);
Psi_pd = V*diag(d)*V';
try
    W_samp = wishrnd(inv(Psi_pd), nu);
    W = inv(W_samp);
catch
    W = Psi_pd/(nu - p - 1 + 1e-8);
end
W = (W+W')/2 + 1e-8*eye(p);
end
