function M = nearest_psd(A, floor_eig)
%NEAREST_PSD  Project a symmetric matrix onto the PSD cone.
%
%   M = utils.nearest_psd(A)               floor_eig = 0
%   M = utils.nearest_psd(A, floor_eig)    lifts eigenvalues to >= floor_eig
%
%   Symmetrizes A and clips eigenvalues below floor_eig.

    if nargin < 2, floor_eig = 0; end
    A = (A + A') / 2;
    [V, D] = eig(A);
    d = diag(D);
    d(d < floor_eig) = floor_eig;
    M = V * diag(d) * V';
    M = (M + M') / 2;
end
