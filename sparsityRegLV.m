function [varargout] = sparsityRegLV(p, res, nf, nFcn)
%
% function [varargout] = sparsityRegLV(p, res, nf, nFcn)
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
%   This function gives the residual vec(A) and its Jacobian or the
%   function, gradient and Hessian values of ||vec(A)|| where A is the
%   interaction matrix of a Lotka-Volterra system.
%
% Input arguments:
%   p       - model parameters for the Lotka-Volterra equations ([r;vec(A)])
%   res     - switch to indicate return (1: r and J; 0: f, g, and H)  
%   nf      - number of state variables in the Lotka-Volterra model
%   nFcn    - function handle of the norm being used
%
% Output arguments:
%   varargout   - either residual and Jacobian or function, gradient, and Hessian
%
% Example:
%   [f, g, H] = sparsityRegLV([1 2 3 4 5 6]', 0, 2, @huberFcn);
%
% References:
%

% only r or f is required
if nargout == 1
    
    % residual accounting for known parameter values and relationships
    r = p(nf+1:end);
    
    % r is required
    if res        
        varargout{1} = r;        
    % f is required
    else        
        f = nFcn(r);        % calculate norm of residual vector
        varargout{1} = f;        
    end
    
% more than r or f is required
else
    
    % residual accounting for known parameter values and relationships
    r = p(nf+1:end);
    
    % Jacobian accounting for known parameter values and relationships
    nr = size(r, 1);                                        % number of rows in residual
    J = [sparse(nr,nf), speye(nr,nr)];    % Jacobian construction
    
    % r and J are required
    if res
        varargout{1} = r;
        varargout{2} = J;
    % more than f is required
    else
        % f and g are required
        if nargout == 2            
            [f, g] = nFcn(r, J);    % calculate norm of residual vector and its gradient         
            varargout{1} = f;
            varargout{2} = g;  
        % f, g, and H are required
        else            
            [f, g, H] = nFcn(r, J); % calculate norm of residual vector and its gradient and Hessian           
            varargout{1} = f;
            varargout{2} = g;
            varargout{3} = H;  
        end        
    end
    
end

end