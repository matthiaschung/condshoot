function [f, g, H] = huberFcn(r, J, param)
%
% [f g H] = huberFnn(r, J, param)
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
%   This function calculates an approximation to the 1-norm using the Huber 
%   function. The function also finds the gradient and Hessian if desired.
%
% Input arguments:
%   r       - residuals
%   J       - Jacobian
%   #param  - further options for huberFcn
%     tol   - l2 approximation on interval (-tol,tol) [default '1e-4']
%
% Output arguments:
%   f   - l1 norm approximation
%   g   - l1 norm approximation of gradient
%   H   - l1 norm approximation of Hessian
%
% Example:
%   f = huberFcn([1 2 3 4 5 6 ]', [], {'tol', 1e-2})
%
% References:
%

% default smoothing parameter
tol = 1e-4;

% overwrite default parameters
if nargin == nargin(mfilename)
    for j = 1:size(param,1)
        eval([param{j,1},'= param{j,2};']);
    end
end

% only f is required
if nargout == 1
    
    % relevant information
    n = length(r);
    rAbs = abs(r);
    
    % index partition
    idx1 = find(rAbs >= tol);   % r =< -tol or r >= tol
    idx2 = find(rAbs < tol);    % -tol < r < tol
    
    % function evaluation
    f = zeros(n, 1);
    f(idx1) = rAbs(idx1) - tol/2;
    f(idx2) = r(idx2).^2/(2*tol);
    f = sum(f);

% more than f is required
elseif nargout > 1
    
    % relevant information
    n = length(r);
    rAbs = abs(r);
    
    % index partition
    idx1 = find(rAbs >= tol);   % r =< -tol or r >= tol
    idx2 = find(rAbs < tol);    % -tol < r < tol
    
    % function evaluation
    f = zeros(n, 1);
    f(idx1) = rAbs(idx1) - tol/2;
    f(idx2) = r(idx2).^2/(2*tol);
    f = sum(f);
    
    % gradient evaluation
    g = zeros(n, 1);
    g(idx1) = sign(r(idx1));
    g(idx2) = r(idx2)/tol;
    g = J'*g;
    
    % Hessian is required
    if nargout == 3
        % Hessian approximation evaluation
        H = zeros(n, 1);
        H(idx2) = 1/tol;
        H = J'*bsxfun(@times, H, J);
    end
    
end

end