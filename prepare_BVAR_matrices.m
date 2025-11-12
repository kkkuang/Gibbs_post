function [Y,X,M,T,KK,K] = prepare_BVAR_matrices(Y,p,constant)
% Consistent layout for all BVAR routines:
%   Y : (T x M) original series
% Returns:
%   Y : (T-p x M) responses
%   X : (T-p x KK) regressors  [const? , Y(t-1) ... Y(t-p)]
%   M : #variables
%   T : effective T after lags
%   KK: #regressors (constant + M*p)
%   K : KK*M (sometimes used downstream)

[Traw,M] = size(Y);
KK = M*p + (constant==1);
K  = KK*M;

% lag stack
Ylag = mlag2(Y,p);           % (Traw x M*p), padded with zeros top-p rows
Ylag = Ylag(p+1:end,:);      % drop first p

% responses and design
Y = Y(p+1:end,:);            % (T-p x M)
if constant
    X = [ones(Traw-p,1), Ylag];
else
    X = Ylag;
end
T = size(Y,1);

end
