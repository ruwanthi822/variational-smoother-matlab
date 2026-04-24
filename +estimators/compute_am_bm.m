function [a_m, b_m] = compute_am_bm(m, p)
%COMPUTE_AM_BM  Second-order-VI scalar coefficients used in the VI^(2)
%               backward pass. These are taken verbatim from
%               Master_updated.m and are the paper-producing values.
%
%   a_m = m * (m*(m - p) - p + 1) / ((m - p) * (m - p - 1) * (m - p - 3))
%   b_m = m * (2*m - p - 1)       / ((m - p) * (m - p - 1) * (m - p - 3))
%
%   These require m > p + 3 for finite second moments.

    if m <= p + 3
        error('compute_am_bm: need m > p + 3 for finite second moments (m=%d, p=%d).', m, p);
    end
    denom = (m - p) * (m - p - 1) * (m - p - 3);
    a_m = m * (m*(m - p) - p + 1) / denom;
    b_m = m * (2*m - p - 1)       / denom;
end
