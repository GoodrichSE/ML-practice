function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

distances = []; % Temporary matrix for holding all centroid distances. Will be m x K.

% for i = 1:size(centroids, 1) % One centroid per row
for i = 1:K % One centroid per row (centroids is a column vector)
	% centroid = centroids(i,:); %size(centroid) == 1 x n
	% % Distance to each example (m x 1 vector)
	% distance = norm((X - centroid), 'fro', 'rows');
	
	distance = norm((X - centroids(i,:)), 'fro', 'rows');
	% Accrue distances.
	distances = [distances distance];
endfor %size(distances) == mxK

[val, idx] = min(distances, [], 2); % Closest distance to centroid per row (i.e. per example)

% =============================================================

end

