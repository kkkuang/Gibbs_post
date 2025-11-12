function [Y_out, meta] = auto_stationarize_matrix(Y_in)
% AUTO_STATIONARIZE_MATRIX
% Make each series approximately stationary by trying, in order:
%   1) level, 2) log(level) if positive, 3) diff(level), 4) diff(log(level)) if positive.
% If any series is differenced, the function drops the first row for ALL series
% to keep alignment and records this choice in the metadata.
%
% INPUT
%   Y_in : T x k matrix
%
% OUTPUT
%   Y_out : T' x k transformed matrix (T' = T or T-1 if any differencing used)
%   meta  : 1 x k struct array with fields:
%           - method: 'level' | 'log' | 'diff' | 'diff_log'
%           - used_test: 'adf' | 'kpss' | 'heuristic'
%           - differenced: 0/1
%           - logged: 0/1
%           - y0: first observed level (for potential back-transform)
%           - drop_first_globally: 0/1 (same for all series)
%
% Notes:
%   Uses adftest and kpsstest if available; otherwise falls back to a
%   robust heuristic based on lag-1 autocorrelation and MAD shrinkage.

    [T, k] = size(Y_in);
    Y_tmp  = NaN(T, k);
    need_diff = false(1,k);
    meta = repmat(struct('method',[],'used_test',[],'differenced',0,'logged',0,'y0',[], ...
                         'drop_first_globally',0), 1, k);

    hasADF  = (exist('adftest','file') == 2);
    hasKPSS = (exist('kpsstest','file') == 2);

    for j = 1:k
        y = Y_in(:,j);
        yallpos = all(y > 0);
        y0 = y(1);

        % Try level
        [is_stat_level, tag_level] = is_stationary_series(y, hasADF, hasKPSS);
        if is_stat_level
            Y_tmp(:,j) = y;
            meta(j) = setmeta('level', tag_level, 0, 0, y0);
            continue;
        end

        % Try log-level if positive
        if yallpos
            [is_stat_log, tag_log] = is_stationary_series(log(y), hasADF, hasKPSS);
            if is_stat_log
                Y_tmp(:,j) = log(y);
                meta(j) = setmeta('log', tag_log, 0, 1, y0);
                continue;
            end
        end

        % Try diff(level)
        dy = diff(y);
        [is_stat_dy, tag_dy] = is_stationary_series(dy, hasADF, hasKPSS);
        if is_stat_dy
            Y_tmp(2:end,j) = dy;
            need_diff(j) = true;
            meta(j) = setmeta('diff', tag_dy, 1, 0, y0);
            continue;
        end

        % Try diff(log(level)) if positive
        if yallpos
            dlogy = diff(log(y));
            [is_stat_dlogy, tag_dlogy] = is_stationary_series(dlogy, hasADF, hasKPSS);
            if is_stat_dlogy
                Y_tmp(2:end,j) = dlogy;
                need_diff(j) = true;
                meta(j) = setmeta('diff_log', tag_dlogy, 1, 1, y0);
                continue;
            end
        end

        % Fallback: diff(level)
        Y_tmp(2:end,j) = dy;
        need_diff(j) = true;
        meta(j) = setmeta('diff', 'heuristic', 1, 0, y0);
    end

    % Explicit global drop if any series differenced
    drop_first = any(need_diff);
    if drop_first
        Y_out = Y_tmp(2:end,:);
    else
        Y_out = Y_tmp;
    end
    for j = 1:k
        meta(j).drop_first_globally = double(drop_first);
    end
end

% ----------------------- helpers -----------------------

function m = setmeta(method, tag, differenced, logged, y0)
    m.method = method;
    m.used_test = tag;
    m.differenced = differenced;
    m.logged = logged;
    m.y0 = y0;
    m.drop_first_globally = 0; % set later
end

function [is_stat, used_tag] = is_stationary_series(x, hasADF, hasKPSS)
% Decide stationarity using ADF if available; else KPSS; else a robust heuristic.
    is_stat = false; used_tag = 'heuristic';
    x = x(:);

    % ADF: H0 = unit root. adftest==true => reject unit root => stationary
    if hasADF
        try
            is_stat = adftest(x);
            used_tag = 'adf';
            return;
        catch
        end
    end

    % KPSS: H0 = stationarity. kpsstest==0 => fail to reject => stationary
    if ~is_stat && hasKPSS
        try
            h = kpsstest(x);
            is_stat = (h == 0);
            used_tag = 'kpss';
            return;
        catch
        end
    end

    % Robust heuristic: high persistence + big MAD drop on differencing => nonstationary
    try
        if numel(x) < 10
            is_stat = false; used_tag = 'heuristic'; return;
        end
        x0 = x - median(x,'omitnan');
        ac1 = lag1_autocorr(x0);
        mad_level = robust_mad(x0);
        dx = diff(x0);
        mad_diff  = robust_mad(dx);
        nonstat_like = (abs(ac1) > 0.9) && (mad_diff < 0.8 * mad_level);
        is_stat = ~nonstat_like;
        used_tag = 'heuristic';
    catch
        is_stat = false; used_tag = 'heuristic';
    end
end

function r1 = lag1_autocorr(x)
    x = x(:);
    x = x - mean(x,'omitnan');
    num = sum(x(2:end).*x(1:end-1),'omitnan');
    den = sum(x.^2,'omitnan');
    r1 = num / max(den, eps);
    if ~isfinite(r1), r1 = 0; end
end

function s = robust_mad(x)
% Robust MAD with Gaussian consistency factor
    x = x(:);
    m = median(x,'omitnan');
    s = median(abs(x - m),'omitnan');
    s = s * 1.4826;             % consistency for N(0,1)
    if ~isfinite(s) || s <= 0
        s = std(x, 0, 1, 'omitnan');
        if ~isfinite(s) || s <= 0, s = 1; end
    end
end
