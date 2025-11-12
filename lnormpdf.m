
function y = lnormpdf(x,mu,sigma)

if nargin<1
    error(message('stats:normpdf:TooFewInputs'));
end
if nargin < 2
    mu = 0;
end
if nargin < 3
    sigma = 1;
end


% Return NaN for out of range parameters.
sigma(sigma <= 0) = NaN;

try
    x = (x-mu)./sigma; 
    y = -0.5*x.*x - log(sigma) - 0.5*log(2*pi);
catch
    error(message('stats:normpdf:InputSizeMismatch'));
end