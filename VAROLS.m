function [beta,sigma] = VAROLS(y,x,M,p,constant)

%% function to estimate VAR using OLS

beta = (x'*x)\(x'*y);
sigma = (y - x*beta)'*(y - x*beta) / max(1, size(y,1) - size(x,2));

B = [beta(constant+1:end,:)'; eye(M*(p-1)) , zeros(M*(p-1),M)];
if max(abs(eig(B)))>1
    warning('parameters of VAR in DGP are not stationary')
end