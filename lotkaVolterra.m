function [f, dfdy, dfdp] = lotkaVolterra(~, y, p)
%
% function [f, dfdy, dfdp] = lotkaVolterra(~, y, p)
%
% Author:
%   (c) Matthias Chung (mcchung@vt.edu)
%       Justin Krueger (kruegej2@vt.edu)
%
% Date: July 2014
%
% MATLAB Version: 8.1.0.604 (R2013a)
%
% Description:
%   The Lotka-Volterra differential equation
%     y' =  diag(y)(r+Ay)
%   with nf species (length of r is nf and dimension of A is nf x nf).
%   A and r are collected in the parameter p = [r; vec(A)].
%
% Input arguments:
%   ~        - time (no time dependence, autonomous system)
%   y        - value at time(s) t
%   p        - parameters (length nf^2+nf)
%
% Output arguments:
%   f        - first output argument is the model function
%                  f = [y_1';...;y_nf']
%   dfdy     - second output argument is df/dy
%   dfdp     - third output argument is df/dp
%
%   f    = [f_1(t_1)  ... f_1(t_nt) ]
%          [  ...     ...    ...    ]
%          [f_nf(t_1) ... f_nf(t_nt)]
%
% Dimension: (nf*nt) x (nf*nt)
%   dfdy = [df/dy(t_1)   0   ...   0]
%          [0 ...     ...     ...  0]
%          [0   ...  0   df/dy(t_nt)]
%     
%   where df/dy(t) = [df_1/dy_1(t)   df_1/dy_2(t)  ... df_1/dy_nf(t) ]
%                    [   ...             ...       ...      ...      ]
%                    [df_nf/dy_1(t)  df_nf/dy_2(t) ... df_nf/dy_nf(t)]
%
% Dimension: (nf*nt) x (np)
%   dfdp = [df/dp(t_1) ]
%          [    ...    ]
%          [df/dp(t_nt)]
%
%   where df/dp(t) = [df_1/dp_1(t)  df_1/dp_2(t)  ... df_1/dp_np(t) ]
%                    [   ...            ...       ...       ...     ]
%                    [df_nf/dp_1(t) df_nf/dp_2(t) ... df_nf/dp_np(t)]
% 
%   and p = [r; vec(A)]
%
% To expand: use objects to calculate dfdp and dfdy.
%
% Example:
%   [f, dfdy, dfdp] = lotkaVolterra(~, [4 6]', [2 2 3 1 -1 2]')
%
% References:
%

% record dimensions of y
[nf, nt] = size(y);

% decompose parameters
r = p(1:nf);                        % intrinsic growth vector
A = reshape(p(nf+1:end), nf, nf);   % interaction matrix

% model function
f = DYDT(y, r, A, nf, nt);

% df/dy required
if nargout > 1
    dfdy = DFDY(y, r, A, nf, nt); % derivative with respect to the state
    % df/dp required
    if nargout > 2
        dfdp = DFDP(y, nf, nt); % derivative with respect to the parameters
    end
end

end

% -------------------------------------------------------------------------
% Lokta-Volterra model f
function f = DYDT(y, r, A, nf, nt)
f = y.*(bsxfun(@times,r,ones(nf,nt)) + A*y);
end

% -------------------------------------------------------------------------
% derivative of f respect to y
function dfdy = DFDY(y, r, A, nf, nt)
m = nf*nt;
dfdy = spdiags(repmat(r,nt,1),0,m,m) + spdiags(y(:),0,m,m)*kron(speye(nt,nt),A) + spdiags(reshape(A*y,m,1),0,m,m);
end

% -------------------------------------------------------------------------
% derivative of f respect to p
function dfdp = DFDP(y, nf, nt)
m = nf*nt;
dfdr = spdiags(y(:),0,m,m)*repmat(speye(nf,nf),nt,1);
dfdA = spdiags(y(:),0,m,m)*kron(y',speye(nf,nf));
dfdp = [dfdr, dfdA];
end