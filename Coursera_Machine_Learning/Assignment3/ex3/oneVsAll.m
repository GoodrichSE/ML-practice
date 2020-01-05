function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Thoughts while I learn
	% m: num of rows in X
	% n: num of cols in X
	% n+1 accounts for x0
	% all_theta has num_labels x n+1 values
	% fmincg produces theta vector for a value of y. Should be put into all_theta
	% all_theta is a collection of thetas with a row for each test case of y
	% y == 2 tests if the value is 2. There are probably multiple training examples where this is the case. We want all of them for our logistic test.
	% y can be a totally different size than num_labels. num_labels is the number of different possible values of y, where y will be as large as the training set.
	% test_theta built by fmin'ing to the partial deriv output vector, eg [1; 0; 0;...] for testing y1


% Setup
options = optimset('GradObj', 'on', 'MaxIter', 50);
test_theta = zeros(n+1,1); % Container for optimized thetas of a single case, including theta_0
initial_theta = zeros(n+1,1); % Zero vector to initiate fmincg

for ind = 1:num_labels
	% Minimize cost function for the classifier of the current index.
	% Comparing y to index (y == ind) sets all non-index values of y to 0, achieving superposition.
	% Also sets all y-values of training samples that output index to 1, to work with sigmoid
	% fprintf('initial: %f\n', size(initial_theta))
	% fprintf('X: %f\n', size(X))
	% fprintf('y: %f\n', size(y))
	% fprintf('output: %f\n', size(test_theta))
	test_theta = fmincg(@(t)(lrCostFunction(t, X, (y == ind), lambda)), initial_theta, options);
	% store optimized values for theta corresponding to current label in a row of all_theta
	
	all_theta(ind,:) = test_theta';
end

% =========================================================================


end
