%% clear and close everything

close all

%% name file for saving results

% file name
fileName = 'exampleDataAndResults.mat';

%% generate data

fprintf('\nGenerating noisy data  ...');

% dimension of model and model parameters
nf = 4;                                                             % number of state dimensions
pTrue = [2 1 0 -3 0 .6 0 .2 -.6 0 .6 .2 0 -.6 0 .2 -.2 -.2 -.2 0]'; % true parameters

% info for ODE solver
t = linspace(0, 10, 20);    % time series for generated data
y0 = [5 4 3 2];             % true initial conditions of state variables

% ODE solution
fcn = @(t, y) lotkaVolterra(t, y, pTrue);   % right-hand side of the ODE
[t, d] = ode23s(fcn, t, y0);                % generate data at times tData

% adjust orientation of data
t = t';
dTrue = d';

% data perturbations
d = dTrue + (betarnd(2,2,size(dTrue))-.5)/5.*dTrue;

% weights for the data
w = 10*bsxfun(@rdivide, ones(size(t)), std(d,0,2));

fprintf(' done.\n');

%% save data with and without noise
fprintf(['Saving data with and without noise to ', fileName ,' ...']);

save(fileName, 'dTrue', 'd');

fprintf(' done.\n');

%% define model, regularization terms, and regularization parameters

fprintf('Setting up model function, regularization terms, and regularization parameters ...');

% model function
modelFcn = @lotkaVolterra;

% regularization terms
reg1 = @(p, res) sparsityRegLV(p, res, nf, @huberFcn);                                    % first regularization term
regFcn = @(p, tau, q, res, alpha, beta) regOrganizer(p, tau, q, res, alpha, beta, reg1);  % regularization organizer

% regularization parameters
lambda = 5.5098;      % lambda value
alpha = 0.04584;      % alpha value(s)

fprintf(' done.\n');

%% use a Monte-Carlo sampling to approximate a global optimization

% number of Monte-Carlo samples
nS = 10;

fprintf('Sampling parameter space %i times ...', nS);

% sample model parameter space
p = bsxfun(@times, 100, lhsdesign(nS, nf^2+nf)'); % sample

fprintf(' done.\n');

%% setup continuous shooting

fprintf('Organizing optional inputs and results storage for continuous shooting ...');

%optional parameters
paramOPT = {'onScreen', 'off'};                                                                                 % optimization options
paramPE = {'optFcn', @gaussNewton; 'regFcn', regFcn; 'lambda', lambda;  'alpha', alpha; 'paramOPT', paramOPT};  % parameter estimation options

% optimization results storage
optResults = cell(1, nS);

fprintf(' done.\n');

%% run continuous shooting

fprintf('Running continuous shooting %i times ...', nS);

tic;
% optimization
for j = 1:nS    
    % optimization algorithm
    [optResults{j}] = continuousShooting(modelFcn, nf, t, d, w, p(:,j), paramPE);
end
time = toc;

fprintf(' done.\nTime per run = %d seconds.\n' , time/nS);

%% find optimal continuous shooting result

fprintf('Finding best result from %i runs ...', nS);

% initial value and index of smallest objection function value
fMinVal = inf;
fMinInd = 0;

% value and index of smallest objection function value
for j = 1:nS
    if optResults{j}.fMin < fMinVal
        fMinVal = optResults{j}.fMin;
        fMinInd = j;
    end
end

fprintf(' done.\n');

%% save results

fprintf(['Saving all results to ', fileName ,' ...']);

% save optimization results
save(fileName, 'time', 'optResults', 'fMinInd', '-append');

fprintf(' done.\n');

%% display results

fprintf('Plotting best result ...');

% define interval
I = [t(1), t(end)]; 

% generate spline results
tt = linspace(I(1), I(2), 200);
ySpline = cubicSpline(linspace(I(1), I(2), size(optResults{fMinInd}.qMin, 2)), optResults{fMinInd}.qMin, tt);

% setup ODE function and solver
odeFcn = @(t, y) modelFcn(t, y, optResults{fMinInd}.pMin);  % define right-hand side
options = odeset('Refine', 10);

% choose solution direction
solType = 'forward'; % options: 'forward', 'backward'

% solve the ODE system
switch solType
    case 'forward'   
        [tODE, yODE] = ode23s(odeFcn, I, optResults{fMinInd}.y0Min, options); % forward solve
    case 'backward'
        odeFcn = @(t, y) -odeFcn(-t, y);                                                        % modify right-hand side for backward solve
        [tODE, yODE] = ode23s(odeFcn, -fliplr(I), optResults{fMinInd}.qMin(:,end), options);    % backward solve
        tODE = -tODE;                                                                           % adjust t vector for plotting
end

% data, spline, and ODE solution plots
figure;                                                                 % new figure
set(gcf, 'color', 'w', 'units', 'inches', 'position', [4, 4, 7.5, 5])   % format figure
colorOrder = get(gca, 'ColorOrder');                                    % save color order for later use

for j = 1:4
    subplot(3, 2, j), box on, hold on;                                  % new subplot
    set(gca, 'FontName', 'Times', 'FontSize', 12);                      % format subplot
    
    plot(t, dTrue(j,:), '.', 'MarkerSize', 20, 'Color', colorOrder(j,:));       % plot data
    plot(tt, ySpline(j,:), ':', 'LineWidth', 2.25, 'Color', colorOrder(j,:));   % plot spline
    plot(tODE, yODE(:,j), 'LineWidth', 1.5, 'Color', colorOrder(j,:));          % plot ODE solution
    
    title(['State ', num2str(j)]);      % add title
    if j > 2
        xlabel('Time');                 % adjust x-axis label
    else
        set(gca, 'XTickLabel', '');     % adjust x-axis label
    end
    if mod(j,2) == 1
        ylabel('Abundance');            % add y-axis label
    end
end
subplot(3,2,[5 6]), hold on;                                                                                        % add legend to empty subplots
set(gca, 'FontName', 'Times', 'FontSize', 12);                                                                      % format subplots
plot(-1, -1, 'k.', 'MarkerSize', 20), plot(-1, -1, 'k:', 'LineWidth', 2.25), plot(-1, -1, 'k', 'LineWidth', 1.5);   % plot fake points
axis([0 1 0 1]), axis off;                                                                                          % turn off plots
legend('Noiseless Data', 'Cubic Spline', 'Numerical Solution', 'Location', get(gca, 'position'));                   % produce legend

fprintf(' done.\n');