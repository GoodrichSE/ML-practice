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

% I'd like to see if there is a way to build this solution in a way that can handle datasets X
% beyond two dimensions. There might be a way with 3D call arrays([examples, clusters, [features]]), but that sounds super complicated
% and there is a good chance that it ends up being slow. I attemt a few paths here.

% I use a lot of intermediate storage variables here, just to help me debug. 

% Split examples into columns, reflecting the cluster it is nearest to
clusterIdx = (idx == 1:K); % m x K, boolean number, with only one 1 per row
	% for idx = 1:K
		% centVec = (centroid == idx);
		% clusterIdx = [clusterIdx centVec];
	% endfor

%%% Path one: build a 3-d array %%%
% (This solution is probably convoluted but is most conceptually clear to me)
% Copy clusterIdx into 3rd dimension, once for each example feature
idxBlock = repmat(clusterIdx, [1; 1; size(X, 2)]); % [examples, cluster indices, example features]
% Matrix approach using reshape/permute, from:
% https://www.mathworks.com/matlabcentral/answers/332898-is-it-possible-to-multiply-a-3d-matrix-with-a-coumn-vector
	% q = permute(idxBlock, [2, 1, 3]); % [clusters, examples, example features]
	% w = reshape
	% assignments = reshape(reshape(permute(A,[2,1,3]),3,[]).'*B,3,[]);
% For-loop approach from the same reference
rs = permute(idxBlock, [1, 3, 2]); % Swap columns and depth [examples, example features, cluster indices]
for i = 1:size(rs, 3)
	assignments(:, :, i) = X .* rs(:, :, i);
endfor
% assignments % sanity check

%%% Path two: Apply the centroids as a mask to x, probably in a cell array %%%
% assignments = cell(size(clusterIdx))
% assignments = cellfun(@(msk) msk.*X, clusterIdx, 'UniformOutput', false) % Outputs cell arrays
% assignments = cellfun(@(data) data.*clusterIdx, X, 'UniformOutput', true) % Outputs a matrix
% This only operates on cell arrays. Need to convert X or clusterIdx to a cell array.

%%% Path three: Apply centroids as a mask to each example, feature-by-feature %%%
% for i = 1:size(X, 2) % Apply each column of the clusterIdx mask to each feature in the examples
	% assignments(:,:,i) = clusterIdx .* X(:,i);
% endfor % assignments (m x K x n) Rows: examples; cols: centroid indices; depth: per-example data (tuples)

%%% Path four: Build a 2x2 cell array of tuples %%%
% reshape X into paired rows then mat2cell(X, dim1, dim2)
% or put x into cell to start with mat2cell(x)

% Column mean, ignoring zeroes (from https://www.mathworks.com/matlabcentral/answers/109549-how-to-calculate-the-average-without-taking-zeros-values)
avgs = sum(assignments,1) ./ sum(assignments ~= 0,1);

centroids = permute(avgs, [3, 2, 1]);

% =============================================================


end

