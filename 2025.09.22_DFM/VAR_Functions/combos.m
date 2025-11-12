function [combos] = combos(A)

[n, p] = size(A);
if n==1, A = A.'; end
% Extract each column into a cell array of vectors
columns = cell(1, p);
for j = 1:p
    columns{j} = A(:, j);
end

% Create Cartesian product using ndgrid
[C{1:p}] = ndgrid(columns{:});

% Convert the grids into a (n^p) × p matrix of combinations
combos = zeros(numel(C{1}), p);
for j = 1:p
    combos(:, j) = C{j}(:);
end