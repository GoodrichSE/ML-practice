function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% Split centroids into columns of 1's and 0's
% for idx = 1:K
	% centVec = (centroid == idx);
	% clusterIdx = [clusterIdx centVec];
% endfor

% Split examples into columns, reflecting the cluster it is nearest to
clusterIdx = (idx == 1:K); % m x K, boolean number, with only one 1 per row

% Path one: build a 3-d array
% for i = 1:size(X, 2)
	% data(:,:,i) = clusterIdx .* X(:,i);
% endfor % Data (m x K x n) Rows: examples; cols: centroid indices; depth: per-example data (tuples)
% Copy clusterIdx into 3rd dimension, once for each example feature
idxBlock = repmat(clusterIdx, [1; 1; size(X, 2)])
assignments = idxBlock * X;
%Try this (with attention on the reshapes): reshape(reshape(permute(A,[2,1,3]),3,[]).'*B,3,[])
for i = 1:size(idxBlock, 3)
	assignments(:, :, i) = X' .* idxBlock(:, :, i)
endfor
%This solution is probbably convoluted but is most conceptually clear to me

% Path two: Apply the centroids as a mask to x, probably in a cell array
% assignments = cell(size(clusterIdx))
% assignments = cellfun(@(msk) msk.*X, clusterIdx, 'UniformOutput', false) % Outputs cell arrays
% assignments = cellfun(@(data) data.*clusterIdx, X, 'UniformOutput', true) % Outputs a matrix
% This only operates on cell arrays. Need to convert X or clusterIdx to a cell array.

% Path three: Apply centroids as a mask to each example, feature-by-feature

% Path four: Build a 2x2 cell array of tuples
% reshape X into paired rows then mat2cell(X, dim1, dim2)
% or put x into cell to start with mat2cell(x)

% Assign examples to a column based on 
% assignments = clusterIdx' * X
% assignments = clusterIdx' .* X

avgs = mean(assignments, 1)

% =============================================================


end

