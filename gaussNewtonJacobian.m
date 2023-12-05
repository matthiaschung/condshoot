function [x, f, info] = gaussNewtonJacobian(fcn, x, param)
%
% function [x, f, info] = gaussNewtonJacobian(fcn, x, param)
%
% Authors:
%   (c) Julianne Chung                   in June 2013         
%       Matthias Chung (e-mail: mcchung@vt.edu)
%
% MATLAB Version: 7.10.0.499 (R2010a)
%
% Description:
%   The Gauss-Newton method for unconstraint optimization problems.
%
% Input arguments:
%   fcn                   - objective function needs to return residual vector r and Jacobian J
%   x                     - initial value
%   #param                - further options of algorithm
%     lineSearchAlgorithm - type of line search algorithm [ {@armijo} | @goldstein | @strongWolfe | @wolfe ]
%                           use lineSearchParameter = {'parameter', value} to pass line search parameters 
%                           to line search algorithm
%     linSolver           - solver of linear system [ {'\'} | 'bicgstab' | 'pcg' ]
%     onScreen            - print to display [ {'off'} | 'iter' | 'final' ]
%     maxIter             - maximal number of iterations [ 100 * dimension ]
%     tol                 - tolerance [ 1e-6 ]
%
% Output arguments:
%   x                     - local minimizer
%   f                     - local minimum
%
% Details:
%   The parameter x returns a local minimizer of the function f.
%   This Gauss Newton method solves numerically the minimization problem
%                    min_x f(x) = 1/2 ||r(x)||^2.
%   The objective function must be in the following form: [f g H] = fcn(x).
%
% Example:
%   [x, f] = gaussNewton( @(x) 0.5*sin(x)^2, 1, {'onScreen', 'iter'; 'tol', 1e-8})
%
% References:
%   [1] Nocedal, J. and Wright, S. J., Numerical Optimization,
%       Springer, New York, 2006.
%   [2] Gill, and Murray, and Wright, Practical Optimization,
%       Academic Press, London, 1992.
%

% initialize default options of algorithm
onScreen = 'off'; tol = 1e-6; maxIter = 100 * numel(x);
lineSearchAlgorithm = @strongWolfe; lineSearchParameter = {}; epsilon = eps;
linSolver = '\'; maxIterLinSolver = 100;

% rewrite default parameters if needed
if nargin == nargin(mfilename)
  for j = 1:size(param,1), eval([param{j,1},'= param{j,2};']); end
end

% display and algorithm info
if strcmp(onScreen, 'iter') || strcmp(onScreen,'final')
  fprintf('\nGauss-Newton Jacobian algorithm (c) Matthias Chung 2013\n');
  if strcmp(onScreen,'final') == 0
    fprintf('\n %-5s %-6s  %-14s %-14s  %-12s\n','iter','evals','function','norm(g)','stop criteria');	
  end
end

% initialize fixed parameters of algorithm
iter = 0; fcnEvals = [1; 1]; xOld = inf; fOld = inf; lineSearchSuccess = 1;
lsFcn = @(x) leastSquaresFcn(fcn,x); % define the least square sub-function
	
% initial evaluation of the objective function in x
 [r, J] = fcn(x); f = 0.5*(r'*r); g = J'*r;

while 1 % main algorithm
  
  if nargout > 2, info.x(:,iter+1) = x; info.f(iter+1) = f; info.fcnEvals(:,iter+1) = fcnEvals; end  

  % stopping criteria
  STOP1 = abs(fOld - f) <= tol * (1 + abs(f));
  STOP2 = norm(xOld - x,'inf') <= sqrt(tol) * (1 + norm(x,'inf'));
  STOP3 = norm(g,'inf') <= nthroot(tol,3) * (1 + abs(f));
  STOP4 = norm(g,'inf') <= epsilon;
  STOP5 = iter > maxIter;
  
  if strcmp(onScreen,'iter') % display
    fprintf('%5d %6d %14.6e %14.6e %4d%1d%1d%1d%1d \n',...
      iter, fcnEvals(1), f, norm(g), STOP1,STOP2,STOP3,STOP4,STOP5 );
  end
  
  if (STOP1 && STOP2 && STOP3) || STOP4 || STOP5 % check stop criteria
    if STOP5
      warning('Matlab:gaussNewton:maxIter',...
        'Maximum number of iterations reached. Return with recent values.')
    end
    break;
  end
  
  iter = iter + 1;
  
  if strcmp(linSolver,'\') % solve linear system
   	s = -J\r; % QR solve of Js = -r
  else
    fhandle = @(x, trans)afun(x,trans,J);
    [s, tmp] = linSolver(fhandle, -r, tol, maxIterLinSolver);
  end
	% calculate step size s by line search algorithm
	[s, fcnEvals, lineSearchSuccess] = lineSearchAlgorithm(lsFcn, x, s, 1, f, g, fcnEvals, lineSearchParameter); 
	if ~lineSearchSuccess, break; end;
  
  % update step
  xOld = x; fOld = f; x = x + s;
  
  % evaluate function at new x
  [r, J] = fcn(x); f = 0.5*(r'*r); g = J'*r; fcnEvals = fcnEvals + 1;
  
end % while

if (strcmp(onScreen,'iter') || strcmp(onScreen,'final')) && ~STOP5 && lineSearchSuccess % display
  fprintf('\nLocal minimizer found. Minimal function value is %1.8e.\n', f);
end

end

% distance function for linesearch function
function [f, g] = leastSquaresFcn(fcn,x)

if nargout == 1
  r = fcn(x); f = 0.5*(r'*r);
else
  [r, J] = fcn(x); f = 0.5*(r'*r); g = J'*r;
end

end