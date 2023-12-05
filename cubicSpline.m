function [s, dsdt, dsdy, dsdydt] = cubicSpline(t, y, tt)
%
% function [s, dsdt, dsdy, dsdydt] = cubicSpline(t, y, tt)
%
% Author:
%   (c) Matthias Chung (mcchung@vt.edu)
%       Justin Krueger (kruegej2@vt.edu)
%
% Date: February 2014
%
% MATLAB Version: 8.1.0.604 (R2013a)
%
% Description:
%   This function calculates the piecewise polynomials (cubic spline) or the values of
%   a cubic spline s with given knots t and corresponding data y(t). If the
%   third argument tt is given this function evaluates the piecewise polynomials
%   at timepoints tt. Not-a-knot boundary condition are used.
%   Additionally this function calculates derivatives ds/dt, ds/dy and d/dy(ds/dt).
%   Note, the derivatives ds/dy and d/dy(ds/dt) are independent of y and for
%   multiple dimensions of y, dsdy and d/dy(ds/dt) are of the form
%     dsdy = [B 0 ... 0 0]
%            [: :     : :]
%            [0 0 ... 0 B]
%   This function just returns just the diagonal block B for efficiency.
%
% Input arguments:
%   t   - knots of spline (row vector 1 x n+1)
%   y   - interpolation values of spline (matrix m x n+1)
%   tt  - evaluation points of spline (row vector 1 x k)
%
% Output arguments:
%   s        - spline s (piecewise polynomial, evaluated at tt if tt not empty)
%   dsdt     - derivative dsdt of s with respect to t (piecewise polynomial, evaluated at tt if tt not empty)
%   dsdy     - derivative dsdy of s with respect to y (piecewise polynomial, evaluated at tt if tt not empty)
%   dsdydt   - derivative dsdydt of s with respect to y and t (piecewise polynomial, evaluated at tt if tt not empty)
%
% Example:
%   t = linspace(0, 2 * pi, 20);
%   y = [cos(t); sin(t)];
%   tt = linspace(0, 2 * pi, 100);
%   [s, dsdt, dsdy, dsdydt] = cubicSpline(t, y); (returns piecewise polynomials)
%   [s, dsdt, dsdy, dsdydt] = cubicSpline(t, y, tt); (returns spline values)
%
% References:
%   [1] Carl DeBoor, A Practical Guide to Splines, Reprint edition, Springer, 1994.
%

% initialize fixed options of algorithm
n = length(t)-1;

% check input data
if n < 1
    error('There must be at least two data points.');
end

% initialize constants
h = diff(t);         % length of interval parts h
m = size(y, 1);       % dimension of y
diffy = diff(y, 1, 2); % difference of interpolation points

if n == 1 % interpolant is a straight line
    
    s = mkpp(t, [diffy/h, y(:,1)], m);
    if nargout > 1
        dsdt = mkpp(t, diffy/h, m);
        if nargout > 2
            dsdy = mkpp(t, [-1, h; 1, 0 ]/h, 2)';
            if nargout == 4
                dsdydt = mkpp(t, [-1, 1]/h, 2)';
            end
        end
    end
    
elseif n == 2 % interpolant is a parabola
    
    a = (diffy(:,2)/h(2) - diffy(:,1)/h(1))/(h(1)+h(2)); % quadratic coefficient
    s = mkpp(t([1,3]), [a, diffy(:,1)/h(1) - a*h(1), y(:,1)], m); % construct spline s
    if nargout > 1 % construct dsdt
        dsdt = mkpp(t([1,3]), [2*a, diffy(:,1)/h(1) - a*h(1)], m);
        if nargout > 2 % construct dsdy
            a = 1/(h(1)+h(2)) * [1/h(1); -1/h(2)-1/h(1); 1/h(2)]; % quadradic coefficient
            dsdy = mkpp(t([1,3]),[a, 1/h(1)*[-1; 1; 0] - h(1)*a, [1; 0; 0] ], 3)';
            if nargout == 4 % construct dsdydt
                dsdydt = mkpp(t([1,3]), [2*a, 1/h(1)*[-1; 1; 0] - h(1)*a], 3)';
            end
        end
    end
    
else % interpolant is a regular spline
    
    % upper diagonal of matrix
    lambda = (h(2:n)./(h(1:n-1)+h(2:n)))';
    
    % initialize tri-diagonal matrix
    A = spdiags([[1-lambda; 0; 0], 2*ones(n+1,1), [0; 0; lambda]], -1:1, n+1, n+1);
    
    % initialize right hand side of linear system
    B = [zeros(m,1), 6*(bsxfun(@rdivide,diffy(:,2:n),(h(1:n-1)+h(2:n)).*h(2:n)) - bsxfun(@rdivide,diffy(:,1:n-1),(h(1:n-1)+h(2:n)).*h(1:n-1))), zeros(m,1)]';
    
    % set not-a-knot boundary conditions
    A(1, 1:3) = [ -1/h(1), 1/h(1)+1/h(2), -1/h(2)];
    A(n+1, n-1:n+1) = [-1/h(n-1), 1/h(n-1)+1/h(n), -1/h(n)];
    
    M =  (A\B)'; % solve for moments M
    H = bsxfun(@times, ones(m,1), h); % distances h for each dimension of y
    
    % generate piecewise polynomial
    s = mkpp(t, [diff(M,1,2)./(6*H), M(:,1:n)/2, diffy./H - (2*M(:,1:n)+M(:,2:n+1)).*H/6, y(:,1:n)], m);
    
    if nargout > 1 % calculate derivative dsdt
        
        dsdt = mkpp(t, [diff(M,1,2)./(2*H), M(:,1:n), diffy./H - (2*M(:,1:n)+M(:,2:n+1)).*H/6], m); % calculate dsdt
        
        if nargout > 2 % calculate derivatives dsdy and dsdydt
            
            % initialize Jacobian of right hand side
            mu = -6./(h(1:n-1).*h(2:n))';
            JB = spdiags([[-mu.*lambda; 0; 0], [0; mu; 0], [0; 0; -mu.*(1-lambda)]], -1:1, n+1, n+1);
            
            % calculate Jacobian of the moments
            JM = full(A\JB);
            
            % initialize constants
            H = bsxfun(@times, h', ones(1,n+1));
            I0 = [sparse(n,1), speye(n)];
            IN = speye(n, n+1);
            
            % calculate coefficients
            Jc = 1./H.*(I0-IN) - H/6.*(2*IN+I0)*JM;
            Jb = 1/2*IN*JM;
            Ja = 1./(6*H).*(I0-IN)*JM;
            
            dsdy = mkpp(t, [Ja; Jb; Jc; full(IN)]', n+1); % calculate dsdy
            
            if nargout == 4 % calculate dsdydt
                dsdydt = mkpp(t, [3*Ja; 2*Jb; Jc]', n+1);
            end
        end
    end
    
end

% evaluate piecewise polynomial at time points tt otherwise return piecwise polynomials
if nargin > 2 % return spline values
    s = ppval(s, tt);
    if nargout > 1
        dsdt = ppval(dsdt, tt);
        if nargout > 2
            dsdy = ppval(dsdy, tt)';
            if nargout == 4
                dsdydt = ppval(dsdydt, tt)';
            end
        end
    end
end

end