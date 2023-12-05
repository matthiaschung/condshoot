function [varargout] = regOrganizer(p, tau, q, res, alpha, beta, varargin)
%
% function [varargout] = regOrganizer(p, tau, q, res, alpha, beta, varargin)
%
% Author:
%   (c) Matthias Chung (mcchung@vt.edu)
%       Justin Krueger (kruegej2@vt.edu)
%
% Date: August 2014
%
% MATLAB Version: 8.1.0.604 (R2013a)
%
% Description:
%   This function evaluates any combination of regularization and returns
%   either the residuals and their Jacobians or the function,
%   gradient, and Hessian values.
%
% Input arguments:
%   p           - model parameter values
%   tau         - time points for spline knots
%   q           - spline parameters
%   res         - switch to indicate return (1: r and J; 0: f, g, and H)
%   alpha       - regularization parameters for model parameter regularization (row vector)
%   beta        - regularization parameters for spline parameter regularization (row vector)
%   varargin    - regularization function(s) that are functions of p and res or tau, q, and res
%
% Output arguments:
%   varargout   - either residual and Jacobian or function, gradient, and Hessian
%
% Example:
%   p = [1 2 3 4 5 6]';
%   tau = linspace(0, 1, 5);
%   q = [2 4 3 1 10; 5 7 9 5 3];
%   res = 0;
%   alpha = 1e-4;
%   beta = 1e-6;
%   reg1 = @(p, res) sparsityRegLV(p, res, nf, @huberFcn);
%   reg2 = @(tau, q, res) smoothingReg(tau, q, res, @twoNormFcn);
%   [f, g, H] = regOrganizer(p, tau, q, res, alpha, beta, reg1, reg2)
%
% References:
%

% number of regularizations
m = nnz(alpha); % number of regularizations on p
n = nnz(beta);  % number of regularizations on q

% number of unknown parameters
np = size(p, 1);        % number of unknown model parameters
nq = numel(q);          % number of unknown spline parameters

% adjust regularization parameters for scaling residual and Jacobian (not
% function, gradient, and Hessian)
if res
    alpha = sqrt(alpha);
    beta = sqrt(beta);
end

% only r or f is required
if nargout == 1
    % only r is required
    if res
        r = [];
        % add one regularization on p at a time
        for j = 1:m
            rTemp = varargin{j}(p, res); % residual of a single regularization on p
            r = [r; alpha(j)*rTemp];            %#ok<*AGROW> % residual concatenated with previous calculations
        end
        % add one regularization on q at a time
        for j = 1:n
            rTemp = varargin{m+j}(tau, q, res); % residual of a single regularization on q
            r = [r; beta(j)*rTemp];             % residual concatenated with previous calculations
        end
        varargout{1} = r;
    % only f is required
    else
        f = 0;
        % add one regularization on p at a time
        for j = 1:m
            fTemp = varargin{j}(p, res); % function value of a single regularization on p
            f = f + alpha(j)*fTemp;             % function value added to previous calculations
        end
        % add one regularization on q at a time
        for j = 1:n
            fTemp = varargin{m+j}(tau, q, res); % function value of a single regularization on q
            f = f + beta(j)*fTemp;              % function value added to previous calculations
        end
        varargout{1} = f;
    end
% more than r or f is required    
else
    % r and J are required
    if res
        r = [];
        J = sparse(0, np+nq);
        % add one regularization on p at a time
        for j = 1:m
            [rTemp, JTemp] = varargin{j}(p, res);        % residual and Jacobian of a single regularization on p
            r = [r; alpha(j)*rTemp];                            % residual concatenated with previous calculations
            J = [J; alpha(j)*JTemp, sparse(size(rTemp,1), nq)]; % Jacobian concatenated with previous calculations
        end
        % add one regularization on q at a time
        for j = 1:n
            [rTemp, JTemp] = varargin{m+j}(tau, q, res);        % residual and Jacobian of a single regularization on q
            r = [r; beta(j)*rTemp];                             % residual concatenated with previous calculations
            J = [J; sparse(size(rTemp,1), np), beta(j)*JTemp]; % Jacobian concatenated with previous calculations
        end
        varargout{1} = r;
        varargout{2} = J;
    % more than f is required
    else
        % f and g are required
        if nargout == 2
            f = 0;
            g = zeros(np+nq, 1);
            % add one regularization on p at a time
            for j = 1:m
                [fTemp, gTemp] = varargin{j}(p, res);    % function and gradient values of a single regularization on p
                f = f + alpha(j)*fTemp;                         % function value added to previous calculations
                g(1:np,1) = g(1:np,1) + alpha(j)*gTemp;       % gradient value added to previous calculations
            end
            % add one regularization on q at a time
            for j = 1:n
                [fTemp, gTemp] = varargin{m+j}(tau, q, res);        % function and gradient values of a single regularization on q
                f = f + beta(j)*fTemp;                              % function value added to previous calculations
                g(np+1:end,1) = g(np+1:end,1) + beta(j)*gTemp;    % gradient value added to previous calculations
            end
            varargout{1} = f;
            varargout{2} = g;
        % f, g, and H are required
        else
            f = 0;
            g = zeros(np+nq, 1);
            H = zeros(np+nq, np+nq);
            % add one regularization on p at a time
            for j = 1:m
                [fTemp, gTemp, HTemp] = varargin{j}(p, res); % function and gradient values of a single regularization on p
                f = f + alpha(j)*fTemp;                             % function value added to previous calculations
                g(1:np,1) = g(1:np,1) + alpha(j)*gTemp;           % gradient value added to previous calculations
                H(1:np,1:np) = H(1:np,1:np) + alpha(j)*HTemp;   % Hessian value added to previous calculations
            end
            % add one regularization on q at a time
            for j = 1:n
                [fTemp, gTemp, HTemp] = varargin{m+j}(tau, q, res);                 % function and gradient values of a single regularization on q
                f = f + beta(j)*fTemp;                                              % function value added to previous calculations
                g(np+1:end,1) = g(np+1:end,1) + beta(j)*gTemp;                    % gradient value added to previous calculations
                H(np+1:end,np+1:end) = H(np+1:end,np+1:end) + beta(j)*HTemp;    % Hessian value added to previous calculations
            end
            varargout{1} = f;
            varargout{2} = g;
            varargout{3} = H;
        end
    end
end

end