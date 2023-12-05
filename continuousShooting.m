function [varargout] = continuousShooting(mFcn, nf, t, d, w, p, paramPE)
%
% function [optResults] = continuousShooting(mFcn, nf, t, d, w, p, paramPE)
%
% Author:
%   (c) Matthias Chung (mcchung@vt.edu)
%       Justin Krueger (kruegej2@vt.edu)
%
% Date: June 2015
%
% MATLAB Version: 8.4.0.150421 (R2014b)
%
% Description:
%   This parameter estimation procedure minimizes
%
%      min_(p,q) f(p,q) = a*||w(prjFcn(s(tau,q,t))-d)||^2 + b*lambda*||s'(tau,q,T)- mFcn(T,s(tau,q,T),p)||^2,
%
%   for a given ODE model function mFcn, parameter set p, a data set (t,d) and
%   a weight matrix w. lambda is a regularization on the accuracy of the
%   model. s is a cubic spline function uniquely determined by the
%   parameters q and knots tau. s(tau,q,t) means spline with parameters
%   q and knots tau evaluated at times t. s' is the time derivative
%   of spline s.
%
%   By default the method uses a Gauss-Newton optimization method and
%   requires the derivatives of f with respect to y and p.
%   Inital guess for y is d. Parameter a is set to 1/nd where nd is the 
%   total number of data points over all states. The parameter b is set to 
%   1/(nT*nf) where nT is the number of sample times T and nf is the 
%   dimension of the model.
%
%   This algorithm requires the function cubicSpline.m and possibly
%   others depending on the outputs requested.
%
% Input arguments:
%   mFcn        - model function of ODE y' = mFcn(t,y,p) of dimension nf
%   nf          - dimension of the model function
%   t           - time points of dimension 1 x n+1 where measurements are taken (or cell)
%   d           - data values at times t with dimension m x n+1 (or cell)
%   w           - weighting matrix for data values with dimension m x n+1 (or cell)
%   p           - inital guess for parameter values
%   #paramPE
%       optFcn   	- optimization function to be used [default @gaussNewtonJacobian]
%       prjFcn      - projection function of model onto observation (data requires derivatives of prjFcn) [default @linearProjection]
%       lambda      - regularization parameter (accuracy of model equations) [default 1]
%       ntau        - number of time points used to create initial guess for spline parameters q [default 50]
%       nT          - number of time points used to discretize model misfit norm [default 100]
%       regFcn      - additional regularization terms [default @regOrganizer]
%       alpha       - regularization parameters for model parameter regularization (row vector) [default 0]
%       beta        - regularization parameters for spline parameterregularization (row vector) [default 0]
%       q           - initial guess of model solution at time points tau [default does not exist]
%       paramOPT    - all typical parameter to be set for the TIA optimization toolbox [default {}]
%
% Output arguments:
%   varargout   
%       {1} - structure containing basic information such as minimizers, minimum objective function, etc.
%
% Example:
%   modelFcn = @lotkaVolterra;
%   nf = 4;
%   t = linspace(0, 2 * pi, 20);
%   d = [cos(t); sin(t)];
%   w = 10*bsxfun(@rdivide, ones(size(times)), std(data,0,2));
%   p = [2 2 3 1 -1 2]';
%   paramPE = {'lambda', 1e-1};
%   [optResults] = continuousShooting(modelFcn, nf, t, d, w, p, paramPE);
%
% References:
%

% convert data to appropriate format
if iscell(t)
    [t, d, w] = rearrangeData(t, d, w); % reshape if non-consistant measuring time points
end

% dimensions of parameter p
np = size(p, 1);

% set default parameters
optFcn = @gaussNewtonJacobian;                      % optimization algorithm
prjFcn = @linearProjection;                         % projection onto data
lambda = 1;                                         % regularization parameter for accuracy of model equations
ntau = 50;                                          % number of knots for spline
nT = 100;                                           % number of evalauation points for model misfit

% set default regularization for model and spline parameters (none)
regFcn = @regOrganizer; % regFcn will return 0's when used
alpha = 0;              % regularization parameter(s) for model regularization
beta = 0;               % regularization parameter(s) for spline regularization

% default options for optimization toolbox
paramOPT = {};

% rewrite default options if needed
if nargin == nargin(mfilename)
    for j = 1:size(paramPE,1)
        eval([paramPE{j,1},'= paramPE{j,2};']);
    end
end

% set outputs of misfit function to be either residual or fuction value
res = 1;
if ~strcmp(func2str(optFcn), 'gaussNewtonJacobian')
    res = 0;
end

% time discretizations
tau = linspace(t(1), t(end), ntau);     % set default knots of the spline
T = linspace(t(1), t(end), nT);         % choose default evaluation points of T for model misfit s'(tau,q,T) - f(T,s(tau,q,T),p)

% inital guess for spline parameters if none is given
if ~exist('q','var')
    q = zeros(nf, ntau);
    % construct q for one state variable at a time
    for j = 1:nf
        idxData = find(d(j,:) ~= 0);
        % ensure at least two data points for spline
        if length(idxData) > 2 
            q(j,:) = cubicSpline(t(idxData), d(j,idxData), tau);    % interpolate data for q values
        end
    end
end

% reshape w and d
w = w(:);
d = d(:);

% remove all indices where no data is available
idxFull = find(w ~= -1);
nidxFull = length(idxFull);

% remove all indices where no data is available and all indices the user wishes to ignore
idx = find(w ~= -1 & w ~= 0);
nidx = length(idx);
w = w(idx)/sqrt(nidxFull); % normalization of residual weights w in s(tau,q,t)-d by sqrt(nd)
d = d(idx);

% pre-compute dsdqD, dsdqM, and dsdqdtM
[~, ~, dsdqD]          = cubicSpline(tau, q, t);    % get spline derivatives for data misfit
[~, ~, dsdqM, dsdqdtM] = cubicSpline(tau, q, T);    % get spline derivatives for model misfit
e = speye(nf, nf);
dSdQD = kron(dsdqD, e);                             % replicate dsdqD for correct model dimension
dSdQM = kron(dsdqM, e);                             % replicate dsdqM for correct model dimension
dSdQdtM = kron(dsdqdtM, e);                         % replicate dsdqdtM for correct model dimension

% objective function
oFcn = @(pq) misfitFcn(pq, mFcn, nf, d, w, np, prjFcn, idx, nidx, lambda, tau, ntau, T, nT, dsdqD, dsdqM, dsdqdtM, dSdQD, dSdQM, dSdQdtM, res, regFcn, alpha, beta);

% initial search parameters pq = (p (unknown),q)
pq = [p; q(:)];

% solve parameter estimation scheme
[pq, f] = optFcn(oFcn, pq, paramOPT);

% extract optimal parameters p and q, approximated state y, and initial condition y0
p = pq(1:np);                               % update optimized parameters
q = reshape(pq(np+1:end), nf, ntau);                   % reshape q so the spline parameter set has the proper dimensions
y = cubicSpline(tau, q, t);                             % construct an approximation of states y using a cubic spline and optimized spline parameters
y0 = y(:, 1);                                           % provide the optimized initial conditions for the states y

% store optimization and decomposition results
varargout{1} = cell2struct({p, q, y0, y, f}, {'pMin', 'qMin', 'y0Min', 'yMin', 'fMin'}, 2);

end

% -------------------------------------------------------------------------
% arrange times, data, and weights in a usuable form
function [t, d, w] = rearrangeData(T, D, W)

% create sorted vector of measuring times and remove duplicate entries
t = unique(sort(cell2mat(T')));

% generate -1 at non measured points in d and w
w = -1*ones(length(T), length(t));
d = -1*ones(length(T), length(t));
for j = 1:length(T)
    for i = 1:length(T{j})
        w(j,T{j}(i) == t) = W{j}(i); % replace -1 any place a weight is given
        d(j,T{j}(i) == t) = D{j}(i); % replace -1 any place a data point is given
    end
end

end

% -------------------------------------------------------------------------
% calculate the objective function value and/or other needed values
function [varargout] = misfitFcn(pq, mFcn, nf, d, w, np, prjFcn, idx, nidx, lambda, tau, ntau, T, nT, dsdqD, dsdqM, dsdqdtM, dSdQD, dSdQM, dSdQdtM, res, regFcn, alpha, beta)

% split search parameter pq into parameters p and q
p = pq(1:np);                               % update optimized parameters
q = reshape(pq(np+1:end), nf, ntau);        % reshape q so the spline parameter set has the proper dimensions

% normalization of residuals s'(T)- f(T,s(tau,q,T),p)
lambda = sqrt(lambda/(nf*nT));

% calculate spline interpolations and derivative of splines
sD = q*dsdqD';
sM = q*dsdqM';
dsdtM = q*dsdqdtM';

% [sD]        = cubicSpline(tau, q, t); % evaluate s(tau,q,t) for data misfit
% [sM, dsdtM] = cubicSpline(tau, q, T); % evaluate s(tau,q,T) for model misfit

% only r or f is required
if nargout == 1
    
    % calculate model function f
    F = mFcn(T, sM, p);
    
    % calculate projection
    prjsD = prjFcn(sD); % calculate projection
    prjsD = prjsD(:);   % vectorize projection
    prjsD = prjsD(idx); % remove indices of unused values from the projection
    
    % residual
    r = [w.*(prjsD-d); lambda*(dsdtM(:)-F(:))]; % combined residual of data and model misfit
    
    % r is required
    if res 
        rR = regFcn(p, tau, q, res, alpha, beta);    % calculate residual for regularization term(s)
        varargout{1} = [r; rR];                             % add regularization residuals
    % f is required
    else
        f = regFcn(p, tau, q, res, alpha, beta); % calculate function value for regularization term(s)
        varargout{1} = 0.5*(r'*r) + f;                  % calculate function
    end

% more than r or f is required
else
    
    % calculate model function f and derivatives fs and fp
    [F, Fs, Fp] = mFcn(T, sM, p);
   
    % calculate projection
    [prjsD, dprjsD] = prjFcn(sD);   % calculate projection and its derivative
    prjsD = prjsD(:);               % vectorize projection
    prjsD = prjsD(idx);             % remove indices of unused values from the projection
    dprjsD = dprjsD(idx,:);         % remove rows of derivatives due to unused values
    
    % residual and Jacobian
    r = [w.*(prjsD-d); lambda*(dsdtM(:)-F(:))];                                                             % combined residual of data and model misfit
    J = [sparse(nidx,np), spdiags(w,0,nidx,nidx)*dprjsD*dSdQD; -lambda*Fp, lambda*(dSdQdtM - Fs*dSdQM)];   % Jacobian of data and model misfit
    
    % r and J are required
    if res 
        [rR, JR] = regFcn(p, tau, q, res, alpha, beta);  % calculate residual and Jacobian for regularization term(s)
        varargout{1} = [r; rR];                                 % add regularization residuals
        varargout{2} = [J; JR];                                 % add regularization Jacobian
    % more than f is required
    else
        % f and g are required
        if nargout == 2
            [f, g] = regFcn(p, tau, q, res, alpha, beta);    % calculate function and gradient values for regularization term(s)
            varargout{1} = 0.5*(r'*r) + f;                          % calculate function
            varargout{2} = J'*r + g;                                % calculate gradient 
        % f, g, and H are required
        else 
            [f, g, H] = regFcn(p, tau, q, res, alpha, beta); % calculate function, gradient, and Hessian values for regularization term(s)
            varargout{1} = 0.5*(r'*r) + f;                          % calculate function 
            varargout{2} = J'*r + g;                                % calculate gradient
            varargout{3} = J'*J + H;                                % Calculate Hessian
        end
    end
    
end

end