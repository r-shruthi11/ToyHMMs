function [bestPerm, new] = permRows(est, mod)
% Permute the rows of MATest to match it with MATmod and return the
% permutation bestPerm that does this

%MATest and MATmod should be the same size

nRows = size(est, 1);

idx = 1:1:nRows;

I = perms(idx);

errs_abs = zeros(size(I, 1), 1);

errs_rms = zeros(size(I, 1), 1);

for i = 1:size(I, 1)
    
    perm = est(I(i, :), :);
    
    errs_abs(i, 1) = sum(sum(abs(perm-mod)));
    
    errs_rms(i,1) = calcRMS(perm, mod);
    
    
end

% [~, minabsE] = min(errs_abs);

[~, minrmsE] = min(errs_rms);

% bestPerm = I(minabsE, :);

bestPerm = I(minrmsE, :);

new = est(bestPerm, :);


    
    
