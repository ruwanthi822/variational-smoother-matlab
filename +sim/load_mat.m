function C = load_mat(cfg)
%LOAD_MAT  Load a ground-truth covariance trajectory from a .mat file.
%
%   C = sim.load_mat(cfg)
%
%   Reads cfg.sim.load_mat (absolute path) and returns the variable named
%   cfg.sim.load_varname (default 'C') cast to shape [T x p x p]. Also
%   updates cfg.sim.T and cfg.sim.p implicitly via main.m (caller).
%
%   Accepted shapes of the stored variable:
%       [T x p x p]   (used directly)
%       [p x p x T]   (permuted to [T p p])
%
%   Typical files: Step_C.mat, Simulation_03_C.mat, Asilomar_C.mat, ...

    if ~isfield(cfg.sim, 'load_mat') || isempty(cfg.sim.load_mat)
        error('sim.load_mat: cfg.sim.load_mat must be set to a .mat file path.');
    end
    if ~exist(cfg.sim.load_mat, 'file')
        error('sim.load_mat: file not found: %s', cfg.sim.load_mat);
    end

    varname = 'C';
    if isfield(cfg.sim, 'load_varname') && ~isempty(cfg.sim.load_varname)
        varname = cfg.sim.load_varname;
    end

    S = load(cfg.sim.load_mat);
    if ~isfield(S, varname)
        names = fieldnames(S);
        error('sim.load_mat: variable "%s" not found in %s. Variables in file: %s', ...
              varname, cfg.sim.load_mat, strjoin(names, ', '));
    end

    C = S.(varname);
    sz = size(C);
    if ndims(C) ~= 3 || sz(2) ~= sz(3)
        % try [p x p x T] -> [T x p x p]
        if ndims(C) == 3 && sz(1) == sz(2)
            C = permute(C, [3 1 2]);
        else
            error('sim.load_mat: unexpected shape %s for %s. Need [T p p] or [p p T].', ...
                  mat2str(sz), varname);
        end
    end
    sz = size(C);

    fprintf('  loaded ground-truth C from %s (T=%d, p=%d)\n', cfg.sim.load_mat, sz(1), sz(2));
end
