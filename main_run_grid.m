% run_grid.m  — scenario sweeper (method × loss), prints compact summaries
close all; clear; clc;

addpath(genpath(fullfile('.', 'Auxiliary_Functions')));
addpath(genpath(fullfile('.', 'VAR_Functions')));

matfile_path = fullfile('DFM','Results','baseline','lag4','DFM_G_ObsShock_1.mat');

% ---- scenario grid ----
tails       = {'gaussian','t','sv','t+sv'};
nus         = [NaN, 5, NaN, 5];           % only used for t / t+sv
cont_types  = [0, 1, 2, 3, 4];            % 0=none, 1=MAO, 2=MIO, 3=MLS, 4=MTC
cont_rates  = [0.05, 0.1];
out_mag     = 5;

% Which method×loss to compare (losses are pruned per method below)
methods = {'ols','robust','bvar-llb','bvar-llb-ssvs', ...
           'bvar-smc','bvar-vb','bvar-mh','lasso', ...
           'sparse-robust-admm','bvar-ald','bvar-ald-l(1/2)'};

losses  = {'ols','huber','lad','studentt'};   % switchable losses (per-method pruning happens below)

Rows = [];
row = 0;

for it = 1:numel(tails)
  for ic = 1:numel(cont_types)
    for ir = 1:numel(cont_rates)
      row = row + 1;

      % --- settings for this scenario ---
      settings = struct();
      settings.simul.T = 200; settings.simul.T_burn = 100;
      settings.simul.n_MC = 100; settings.simul.seed = 1000 + row;
      settings.est.with_IV = 0;
      settings.est.n_lags_fix = 4;
      settings.est.shock_weight_strategy = 'first_factor';
      settings.specifications.random_n_var = 3;
      settings.specifications.random_fixed_var = [];
      settings.specifications.random_fixed_pos = 1;

      settings.simul.innov = struct('type', tails{it}, 'nu', nus(it), ...
                                    'sv_phi', 0.97, 'sv_sig', 0.2, ...
                                    'sv_mode','common');

      % --- opts used by run_mc_scenario / apply_outliers_and_estimate_VAR ---
      opts = struct();
      opts.p = 12; opts.cons = 1;

      if cont_types(ic) == 0
          opts.outlier_type = 1;           % consistent “no outliers” branch
          opts.contamination_rate = 0;
      else
          opts.outlier_type = cont_types(ic);
          opts.contamination_rate = cont_rates(ir);
      end
      opts.outlier_magnitude = out_mag;

      % robust loss plumbing (if you calibrate any deltas/betas, do it inside estimators)
      opts.delta       = 0.7;
      opts.scale_mode  = 'resid';

      % >>> SINGLE SOURCE OF TRUTH for method×loss grid <<<
      opts.estimators = make_estimator_grid(methods, losses);

      % --- run one scenario (no save, quiet) ---
      res = run_mc_scenario(matfile_path, settings, opts, 'DoSave', false, 'Verbose', false);

      % ---- summarize: average 1-step MSE ----
      est = res.estimator_names;                % aligned to columns
      avgMSE1 = mean(res.predMSE1, 1, 'omitnan');   % 1×E

      % ---- average rank by forecast MSE (1=best) ----
      ranks_mse = NaN(size(res.predMSE1));
      for i = 1:size(ranks_mse,1)
          [~,ord] = sort(res.predMSE1(i,:), 'ascend');
          ranks_mse(i,ord) = 1:numel(ord);
      end
      avgRank_mse = mean(ranks_mse, 1, 'omitnan');

      % ---- estimation error (relative Frobenius): ||β̂-β||F / ||β||F ----
      avgEstErr   = mean(res.betaErrRel, 1, 'omitnan');  % 1×E

      % ---- average rank by estimation error (1=best) ----
      ranks_est = NaN(size(res.betaErrRel));
      for i = 1:size(ranks_est,1)
          [~,ord] = sort(res.betaErrRel(i,:), 'ascend');
          ranks_est(i,ord) = 1:numel(ord);
      end
      avgRank_est = mean(ranks_est, 1, 'omitnan');

      % ---- OLS win rates (share of MC reps with lower error than OLS) ----
      ols_idx = find_ols_index(est);
      win_mse = nan(1, numel(est));
      win_est = nan(1, numel(est));
      if ~isempty(ols_idx)
          P = res.predMSE1;       % N×E
          B = res.betaErrRel;     % N×E
          validM = isfinite(P(:,ols_idx));
          validB = isfinite(B(:,ols_idx));
          for j = 1:numel(est)
              vm = validM & isfinite(P(:,j));
              vb = validB & isfinite(B(:,j));
              if any(vm)
                  win_mse(j) = mean(P(vm,j) < P(vm,ols_idx));
              end
              if any(vb)
                  win_est(j) = mean(B(vb,j) < B(vb,ols_idx));
              end
          end
      end

      % ---- pack a tidy row into a struct (safe variable names) ----
      Row = struct();
      Row.tail = string(tails{it});
      Row.outlier_type = cont_types(ic);
      Row.cont_rate    = cont_rates(ir);

      for j = 1:numel(est)
          nm = matlab.lang.makeValidName(est{j});  % e.g. 'ROBUST-HUBER' -> 'ROBUST_HUBER'
          % Forecast MSE summaries
          Row.([nm '_MSE1'])      = avgMSE1(j);
          Row.([nm '_Rank_MSE'])  = avgRank_mse(j);
          Row.([nm '_Win_MSE'])   = win_mse(j);
          % Estimation error summaries
          Row.([nm '_EstErr'])    = avgEstErr(j);
          Row.([nm '_Rank_Est'])  = avgRank_est(j);
          Row.([nm '_Win_Est'])   = win_est(j);
      end

      Rows = [Rows; Row]; %#ok<AGROW>
    end
  end
end

Results = struct2table(Rows);

% ---- split Results into compact tables and print ----
metaCols = {'tail','outlier_type','cont_rate'};
metaCols = intersect(metaCols, Results.Properties.VariableNames, 'stable');
allNames = Results.Properties.VariableNames;

TblMSE1    = Results(:, [metaCols, allNames( endsWith(allNames, '_MSE1') )]);
TblRankMSE = Results(:, [metaCols, allNames( endsWith(allNames, '_Rank_MSE') )]);
TblWinMSE  = Results(:, [metaCols, allNames( endsWith(allNames, '_Win_MSE') )]);

TblEstErr    = Results(:, [metaCols, allNames( endsWith(allNames, '_EstErr') )]);
TblRankEst   = Results(:, [metaCols, allNames( endsWith(allNames, '_Rank_Est') )]);
TblWinEst    = Results(:, [metaCols, allNames( endsWith(allNames, '_Win_Est') )]);

fprintf('\n=== Average 1-step MSE (by scenario) ===\n');                  disp(TblMSE1);
fprintf('\n=== Average Rank by MSE (1 = best) ===\n');                   disp(TblRankMSE);
fprintf('\n=== OLS Win-Rate by MSE (share lower than OLS) ===\n');       disp(TblWinMSE);

fprintf('\n=== Average Estimation Error ||β̂-β||F / ||β||F ===\n');       disp(TblEstErr);
fprintf('\n=== Average Rank by Estimation Error (1 = best) ===\n');      disp(TblRankEst);
fprintf('\n=== OLS Win-Rate by Estimation Error ===\n');                 disp(TblWinEst);

% ---- optional: head-to-head rank comparisons (both metrics) ----
pairs = struct( ...
    'A', {'ROBUST_HUBER',          'BVAR_LLB_HUBER',          'BVAR_LLB_SSVS_HUBER'}, ...
    'B', {'ROBUST_OLS',            'BVAR_LLB_OLS',            'BVAR_LLB_SSVS_OLS'} ...
);
compare_pairs_rank(Results, pairs, '_Rank_MSE',  'MSE');
compare_pairs_rank(Results, pairs, '_Rank_Est',  'Estimation');

% ---- helper: build method×loss spec ----
function specs = make_estimator_grid(methods_in, losses_in)
    specs = struct('id',{},'loss',{},'label',{});
    k = 0;
    for i = 1:numel(methods_in)
        id = lower(strtrim(methods_in{i}));
        Lset = losses_in;
        switch id
            case {'ols','lasso','robust sparse','bvar-ald','bvar-ald-l(1/2)'}
                Lset = {'ols'};   % fixed objective
        end
        for j = 1:numel(Lset)
            L = lower(strtrim(Lset{j}));
            k = k + 1;
            specs(k).id    = id;
            specs(k).loss  = L;
            specs(k).label = pretty_label(id, L);
        end
    end
end

function lbl = pretty_label(id, loss)
    IDU = upper(strrep(id,' ','_'));
    switch id
        case 'ols'
            lbl = 'OLS';
        case {'lasso','robust sparse','bvar-ald','bvar-ald-l(1/2)'}
            lbl = IDU;       % ignore loss
        otherwise
            if strcmpi(loss,'ols')
                lbl = sprintf('%s-OLS', IDU);
            else
                L = upper(loss); L = strrep(L,'(1/2)','L1/2');
                lbl = sprintf('%s-%s', IDU, L);
            end
    end
end

function idx = find_ols_index(names)
    idx = [];
    for j = 1:numel(names)
        if startsWith(upper(string(names{j})), "OLS")
            idx = j; return;
        end
    end
end

function compare_pairs_rank(Results, pairs, suffix, tag)
    for k = 1:numel(pairs)
        Acol = [pairs(k).A, suffix];
        Bcol = [pairs(k).B, suffix];
        if all(ismember({Acol,Bcol}, Results.Properties.VariableNames))
            rA = Results.(Acol); rB = Results.(Bcol);
            valid = isfinite(rA) & isfinite(rB);
            winsA = sum(rA(valid) < rB(valid));
            winsB = sum(rB(valid) < rA(valid));
            ties  = sum(rA(valid) == rB(valid));
            N     = sum(valid);
            avgDiff = mean(rB(valid) - rA(valid));  % + => A better (lower rank)
            fprintf('\n[%s] %s vs %s\n', tag, Acol, Bcol);
            fprintf('  A wins: %d   B wins: %d   ties: %d   (out of %d)\n', winsA, winsB, ties, N);
            fprintf('  Mean rank diff (B - A): %.3f  (positive favors A)\n', avgDiff);
        else
            fprintf('\n[skip] Missing columns for pair: %s vs %s (%s)\n', Acol, Bcol, tag);
        end
    end
end
