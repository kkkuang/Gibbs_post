function res = run_mc_scenario(matfile_path, settings, opts, varargin)
% RUN_MC_SCENARIO  Monte Carlo from estimated DFM (ABCD) — single DGP per MC (no IV)

% ---------- args ----------
ip = inputParser;
ip.addParameter('DoSave', true, @islogical);
ip.addParameter('Verbose', true, @islogical);
ip.parse(varargin{:});
DoSave  = ip.Results.DoSave;
Verbose = ip.Results.Verbose;

% ---------- paths ----------
addpath(genpath(fullfile('.', 'Auxiliary_Functions')));
addpath(genpath(fullfile('.', 'Estimation_Routines')));
addpath(genpath(fullfile('.', 'DFM', 'Subroutines')));
addpath(genpath(fullfile('.', 'VAR_Functions')));

% ---------- RNG single source of truth ----------
masterSeed = settings.simul.seed;
rng(masterSeed + 424242, 'twister');   % client-side

% ---------- Load model ----------
if Verbose, fprintf('Loading DFM from: %s\n', matfile_path); end
S = load(matfile_path);
if ~isfield(S, 'DF_model'), error('The MAT file does not contain DF_model.'); end
DF_model = S.DF_model;

if isfield(DF_model,'ABCD') && all(isfield(DF_model.ABCD, {'A','B','C','D'}))
    ABCD = DF_model.ABCD;
else
    if Verbose, fprintf('ABCD not found; constructing via ABCD_fun_DFM.m ...\n'); end
    ABCD = ABCD_fun_DFM(DF_model);
end
if isfield(DF_model,'n_y'),   n_y  = DF_model.n_y;  else, n_y  = size(ABCD.D, 2); end
if isfield(DF_model,'n_fac'), n_fac = DF_model.n_fac; else, n_fac = size(ABCD.B, 2) - n_y; end

model = struct(); model.ABCD = ABCD; model.n_y = n_y;

% shock weight
switch lower(settings.est.shock_weight_strategy)
    case 'first_factor'
        settings.est.shock_weight = [1; zeros(n_fac-1 + n_y, 1)];
    case 'custom'
        if ~isfield(settings.est,'shock_weight_custom') || isempty(settings.est.shock_weight_custom)
            error('Provide settings.est.shock_weight_custom for custom strategy.');
        end
        settings.est.shock_weight = settings.est.shock_weight_custom(:);
    otherwise
        error('Unknown shock_weight_strategy: %s', settings.est.shock_weight_strategy);
end

% ---------- Single DGP selection ----------
k_each = settings.specifications.random_n_var;
pool = 1:n_y;
if ~isempty(settings.specifications.random_fixed_var)
    fixed = settings.specifications.random_fixed_var;
    pool(pool==fixed) = [];
    draw_rest = randsample(pool, k_each-1);
    sel = [fixed, draw_rest];
    pos = settings.specifications.random_fixed_pos;
    sel = sel(:)';
    if pos <= k_each
        sel = [sel(sel~=fixed), fixed];
        if pos < k_each
            sel = [sel(1:pos-1), fixed, sel(pos:end-1)];
        end
    end
else
    sel = randsample(pool, k_each);
end
settings.specifications.var_select = sel(:)';  % 1×k_each
settings.specifications.n_spec     = 1;

% ---------- Build estimator grid (new API or legacy) ----------
if ~isfield(opts,'estimators') && isfield(opts,'estimator_names')
    opts.estimators = legacy_names_to_specs(opts.estimator_names);
end
assert(isfield(opts,'estimators') && ~isempty(opts.estimators), ...
    'Provide opts.estimators (array of {id,loss,label}).');

% ---------- Pilot / preallocate ----------
data_sim_all0 = generate_data(model, settings);
data_sim0     = select_data_fn(data_sim_all0, settings, 1);

pilot_out     = apply_outliers_and_estimate_VAR(data_sim0.data_y, opts);
[K,n,~]       = size(pilot_out.beta_est);
N             = settings.simul.n_MC;

estimator_names = pilot_out.estimator_names(:).';
n_estimators    = numel(estimator_names);

mse_all       = NaN(N, n_estimators);
beta_true_all = NaN(K, n, N);
beta_est_all  = NaN(K, n, n_estimators, N);
transform_meta_all = cell(N,1);

% forecast score containers
predMSE1  = NaN(N, n_estimators);  predMAE1  = NaN(N, n_estimators);
predCRPS1 = NaN(N, n_estimators);  predLPS1  = NaN(N, n_estimators);
predMSE4  = NaN(N, n_estimators);  predMAE4  = NaN(N, n_estimators);
predMSE8  = NaN(N, n_estimators);  predMAE8  = NaN(N, n_estimators);
loss_seq_mse1 = cell(N, n_estimators);
loss_seq_mae1 = cell(N, n_estimators);

% NEW: estimation error metric per MC × estimator
betaErrRel = NaN(N, n_estimators);   % ||β̂-β||F / ||β||F

if Verbose, fprintf('\nStarting Monte Carlo: %d reps\n', N); end

parfor i_mc = 1:N
    rng(double(masterSeed) + i_mc, 'twister');

    data_sim_all = generate_data(model, settings);
    data_sim     = select_data_fn(data_sim_all, settings, 1);
    Yraw         = data_sim.data_y;

    % stationarize consistently across series
    [Y, transform_meta] = auto_stationarize_matrix(Yraw);

    % estimation (method×loss grid)
    out = apply_outliers_and_estimate_VAR(Y, opts);

    mse_all(i_mc, :)         = out.mse;
    beta_true_all(:,:,i_mc)  = out.beta_true;
    beta_est_all(:,:,:,i_mc) = out.beta_est;

    % forecast scores (1, 4, 8)
    [scores, loss_seq] = compute_forecast_scores(Y, out.beta_est, opts.p, opts.cons);
    predMSE1(i_mc, :)  = scores.mse1;   predMAE1(i_mc, :)  = scores.mae1;
    predCRPS1(i_mc, :) = scores.crps1;  predLPS1(i_mc, :)  = scores.lps1;
    predMSE4(i_mc, :)  = scores.mse4;   predMAE4(i_mc, :)  = scores.mae4;
    predMSE8(i_mc, :)  = scores.mse8;   predMAE8(i_mc, :)  = scores.mae8;

    transform_meta_all{i_mc} = transform_meta;

    % store per-estimator loss sequences for potential diagnostics
    for e = 1:n_estimators
        loss_seq_mse1{i_mc, e} = loss_seq.mse1{e};
        loss_seq_mae1{i_mc, e} = loss_seq.mae1{e};
    end

    % NEW: estimation error metric (relative Frobenius)
    Btrue = out.beta_true;
    denom = max(norm(Btrue, 'fro'), 1e-12);
    erow  = NaN(1,n_estimators);
    for e = 1:n_estimators
        Be = out.beta_est(:,:,e);
        if ~all(isnan(Be(:)))
            erow(e) = norm(Be - Btrue, 'fro') / denom;
        end
    end
    betaErrRel(i_mc,:) = erow;
end

if Verbose, fprintf('\nMonte Carlo DONE. Now summarizing...\n'); end

% ---------- optional prints ----------
if Verbose
    fprintf('%-26s %10s %10s %10s %10s\n', 'Estimator', 'Mean MSE', 'Std MSE', 'Min MSE', 'Max MSE');
    fprintf('%s\n', repmat('-', 1, 80));
    for j = 1:n_estimators
        v = mse_all(:, j); v = v(~isnan(v));
        if isempty(v)
            fprintf('%-26s %10s %10s %10s %10s\n', estimator_names{j}, 'NaN','NaN','NaN','NaN');
        else
            fprintf('%-26s %10.6f %10.6f %10.6f %10.6f\n', ...
                estimator_names{j}, mean(v), std(v), min(v), max(v));
        end
    end
end

% ---------- Save (figure + .mat) ----------
if DoSave
    figure('Name','MSE Comparison');
    valid_rows = ~all(isnan(mse_all),2);
    if any(valid_rows)
        boxplot(mse_all(valid_rows, :), 'Labels', estimator_names);
        ylabel('MSE'); grid on;
        switch opts.outlier_type
            case 1, otxt = 'MAO (Additive)';
            case 2, otxt = 'MIO (Innovational)';
            case 3, otxt = 'MLS (Level Shift)';
            case 4, otxt = 'MTC (Temporary Change)';
            otherwise, otxt = 'Unknown';
        end
        title(sprintf('MSE Comparison – %s, contamination=%.1f%%, mag=%g\\sigma', ...
            otxt, 100*opts.contamination_rate, opts.outlier_magnitude));
    end
    run_tag  = sprintf('p%d_type%d_cont%03d_mag%g_T%d', ...
        opts.p, opts.outlier_type, round(1000*opts.contamination_rate), ...
        opts.outlier_magnitude, settings.simul.T);
    savefig(gcf, ['MSE_Box_' run_tag '.fig']);

    save_fname = sprintf('MC_Summary_singleDGP_noIV_p%d_type%d_cont%03d_mag%g.mat', ...
        opts.p, opts.outlier_type, round(1000*opts.contamination_rate), opts.outlier_magnitude);
end

% ---------- Package result ----------
res = struct();
res.mse_all = mse_all;
res.beta_true_all = beta_true_all;
res.beta_est_all  = beta_est_all;
res.estimator_names = estimator_names;
res.opts = opts;
res.settings = settings;
res.transform_meta_all = transform_meta_all;

res.predMSE1 = predMSE1; res.predMAE1 = predMAE1;
res.predCRPS1 = predCRPS1; res.predLPS1 = predLPS1;
res.predMSE4 = predMSE4; res.predMAE4 = predMAE4;
res.predMSE8 = predMSE8; res.predMAE8 = predMAE8;

res.loss_seq_mse1 = loss_seq_mse1; res.loss_seq_mae1 = loss_seq_mae1;

% NEW: return estimation error metric
res.betaErrRel = betaErrRel;

if DoSave
    save(save_fname, '-struct', 'res');
    if Verbose, fprintf('Saved summary to %s\n', save_fname); end
end
end

% ======================== helpers ========================
function specs = legacy_names_to_specs(name_list)
LKNOWN = ["ols","huber","lad","studentt","lms"];
specs = struct('id',{},'loss',{},'label',{});
for i = 1:numel(name_list)
    raw = string(name_list{i});
    parts = split(raw, "-");
    if numel(parts) >= 2 && any(strcmpi(parts(end), LKNOWN))
        loss = lower(parts(end));
        id   = lower(join(parts(1:end-1), "-"));
    else
        id   = lower(raw);
        loss = 'ols';
    end
    specs(end+1) = struct('id', char(id), 'loss', char(loss), 'label', char(raw)); %#ok<AGROW>
end
end
