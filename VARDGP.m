function [X] = VARDGP(beta,sigma,T,n,p,constant)
% Simulate from a simple VAR(p)

X = rand(T+100,n);
for t = p+1:T+100
    Xlag = mlag2(X(1:t,:),p);
    X(t,:) = [ones(1,constant==1) Xlag(t,:)]*beta  + randn(1,n)*chol(sigma);
end
X = X(end-T+1:end,:);