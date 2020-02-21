function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% I'm getting numbers that pass the tests, but I might still be doing this wrong. Should sim be
% a vector of size m? It seems like we're operating on a single example at a time here.

% ||x1 - x2||^2 = (x1_1 - x2_1)^2 +(x1_1 - x2_2)^2 + ... 
% q = sum((x1-x2).^2) / (2 * (sigma.^2));
q = sum((x1-x2).^2) / (2 * (sigma^2));
sim = exp(-q);

% =============================================================
    
end
