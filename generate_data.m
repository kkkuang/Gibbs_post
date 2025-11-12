function data_sim = generate_data(model, settings)
% Generate simulated data from ABCD state-space:
%   s_t = A s_{t-1} + B e_t
%   y_t = C s_{t-1} + D e_t
% External IV: z_t = rho z_{t-1} + alpha * (shock_weight' * e_t) + v_t
%
% NOTE: Non-Gaussian innovations (t, SV, t+SV) are handled in the block
%       where shocks e_t are drawn.

% unpack settings
T      = settings.simul.T;
T_burn = settings.simul.T_burn;

A = model.ABCD.A;
B = model.ABCD.B;
C = model.ABCD.C;
D = model.ABCD.D;

[n_s, n_e] = size(B);

with_IV = settings.est.with_IV;

if with_IV == 1
    rho_grid     = model.IV.rho_grid;
    alpha        = model.IV.alpha;
    sigma_v_grid = model.IV.sigma_v_grid;
else
    rho_grid     = 0.1;
    alpha        = 1;
    sigma_v_grid = 1;
end

shock_weight = settings.est.shock_weight;

% -------------------- draw shocks e_t (this is the only changed part) ----
T_tot = T_burn + T;

% Defaults if 'innov' not provided
innov = struct('type','gaussian','nu',5,'sv_phi',0.97,'sv_sig',0.2,'sv_mode','common');
if isfield(settings,'simul') && isfield(settings.simul,'innov')
    f = fieldnames(settings.simul.innov);
    for i=1:numel(f), innov.(f{i}) = settings.simul.innov.(f{i}); end
end

% Optional covariance of e_t (default identity)
if isfield(settings.simul,'Sigma_e')
    Sigma_e = settings.simul.Sigma_e;
else
    Sigma_e = eye(n_e);
end
% Cholesky for correlation
L = chol((Sigma_e + Sigma_e')/2 + 1e-12*eye(n_e), 'lower');

% Base standard normals
U = randn(T_tot, n_e);   % iid N(0,1)

switch lower(innov.type)
    case 'gaussian'
        E = U;

    case 't'
        nu = innov.nu;
        % Scale-mixture: Z ./ sqrt(chi2/nu). Rescale to unit variance.
        g  = chi2rnd(max(nu,2.1), T_tot, 1) / max(nu,2.1);
        sT = sqrt((max(nu,2.1)-2)./max(nu,2.1)) ./ sqrt(g);   % E[sT.^2] ≈ 1
        E  = bsxfun(@times, sT, U);

    case 'sv'
        phi   = innov.sv_phi;
        sig_h = innov.sv_sig;
        if strcmpi(innov.sv_mode,'indep')
            % independent AR(1) log-vol per shock dimension
            h = zeros(T_tot, n_e);
            v = sig_h * randn(T_tot, n_e);
            % stationary mean so that E[exp(h)] = 1  => mu = -0.5*Var(h)
            var_h = sig_h^2 / max(1 - phi^2, 1e-8);
            mu = -0.5 * var_h;
            h(1,:) = mu;
            for t=2:T_tot, h(t,:) = mu + phi*(h(t-1,:) - mu) + v(t,:); end
            sSV = exp(0.5 * h); % T_tot x n_e
        else
            % single common log-vol factor
            var_h = sig_h^2 / max(1 - phi^2, 1e-8);
            mu = -0.5 * var_h;
            h = zeros(T_tot,1);  h(1) = mu;
            for t=2:T_tot, h(t) = mu + phi*(h(t-1) - mu) + sig_h*randn; end
            sSV = exp(0.5 * h); % T_tot x 1
        end
        E = bsxfun(@times, sSV, U);

    case 't+sv'
        % Student-t scale (unit-variance on average)
        nu = innov.nu;
        g  = chi2rnd(max(nu,2.1), T_tot, 1) / max(nu,2.1);
        sT = sqrt((max(nu,2.1)-2)./max(nu,2.1)) ./ sqrt(g);
        % SV scale with E[exp(h)] = 1
        phi   = innov.sv_phi;
        sig_h = innov.sv_sig;
        if strcmpi(innov.sv_mode,'indep')
            h = zeros(T_tot, n_e);
            v = sig_h * randn(T_tot, n_e);
            var_h = sig_h^2 / max(1 - phi^2, 1e-8);
            mu = -0.5 * var_h;
            h(1,:) = mu;
            for t=2:T_tot, h(t,:) = mu + phi*(h(t-1,:) - mu) + v(t,:); end
            sSV = exp(0.5 * h);
        else
            var_h = sig_h^2 / max(1 - phi^2, 1e-8);
            mu = -0.5 * var_h;
            h = zeros(T_tot,1);  h(1) = mu;
            for t=2:T_tot, h(t) = mu + phi*(h(t-1) - mu) + sig_h*randn; end
            sSV = exp(0.5 * h);  % T_tot x 1
        end
        E = bsxfun(@times, sSV, bsxfun(@times, sT, U));

    otherwise
        error('Unknown settings.simul.innov.type: %s', innov.type);
end

% Impose desired covariance of e_t
data_e = E * L';   % (T_tot x n_e), each row is e_t'

% -------------------- simulate states & observables -----------------------
s = zeros(n_s,1);
data_s = NaN(T_tot, n_s);
for t = 1:T_tot
    s = A * s + B * data_e(t,:)';
    data_s(t,:) = s';
end

% y_t depends on s_{t-1} and e_t (timing consistent with your original code)
data_y = data_s(T_burn:end-1,:)*C' + data_e(T_burn+1:end,:)*D';

% -------------------- simulate IV(s) --------------------------------------
z = NaN(T_tot, length(rho_grid), length(sigma_v_grid));
iv_shock = data_e * shock_weight;   % T_tot x 1
for idx = 1:length(rho_grid)
    for jdx = 1:length(sigma_v_grid)
        z(:,idx,jdx) = filter(1, [1 -rho_grid(idx)], alpha * iv_shock + ...
                               sigma_v_grid(jdx) * randn(T_tot,1));
    end
end
data_z = z(T_burn+1:end,:,:);

% -------------------- collect results -------------------------------------
data_sim.data_y     = data_y;
data_sim.data_shock = data_e((T_burn+1):(T_burn+T), :) * shock_weight;
data_sim.data_z     = data_z;

end
    