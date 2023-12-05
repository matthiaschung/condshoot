function [f, df] = linearProjection(y, C)
%
% function [f, df] = linearProjection(y, C)
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
%   This function maps the state variable y using the mapping defined by C.
%   Additionally, the derivative df/dy can be calculated if desired.
%
% Input arguments:
%   y   - state variable values
%   C   - projection matrix
%
% Output arguments:
%   f   - first output argument is the mapped state variable
%   df  - second output argument is the derivative df/dy
%
% Example:
%   [f, df] = linearProjection([4 5; 6 7; 8 9], [1 1 1])
%
%   f  = [f_1]
%        [...]
%        [f_n]
%
%   df = [ df_1/dy_1  df_1/dy_2 ... df_1/dy_n]
%        [    ...         ...   ...    ...   ]
%        [ df_n/dy_1  df_n/dy_2 ... df_n/dy_n]
%
% References:
%

% only state values are given
if nargin == 1
    % identity mapping
    f = y;  
    % derivative required
    if nargout > 1
        % derivative df/dy
        df = speye(numel(y));   
    end
% projection is given    
else
    % mapping
    f = C*y;
    % derivative required
    if nargout > 1 
        % derivative df/dy
        df = kron(speye(size(f,2)), C); 
    end    
end

end