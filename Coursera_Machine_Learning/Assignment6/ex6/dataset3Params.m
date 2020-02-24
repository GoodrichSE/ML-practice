function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

attempts = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
% Arbitrarily high number that we hope to beat
least_error = 10000;

for i = 1:length(attempts)
	C_attempt = attempts(i);
	for j = 1:length(attempts)
		sigma_attempt = attempts(j);
		
% This is more readable, but passes the whole vector. There's probably a way to make it work.
% for C_attempt = attempts
	% for sigma_attempt = attempts
		
		% The extra course resources told me to use this syntax, which I don't fully understand,
		% from https://www.coursera.org/learn/machine-learning/discussions/weeks/7/threads/ytrutj_YEeai1RIqHM9jYQ:
		model = svmTrain(X, y, C_attempt, @(x1, x2) gaussianKernel(x1, x2, sigma_attempt));
		
		hypothesis = svmPredict(model, Xval);
		
		error = mean(double(hypothesis ~= yval));
		%svmPredict(svmTrain(Xval, y, C_attempt, @(x1, x2) gaussianKernel(X, y, sigma_attempt)), Xval)
		if error < least_error
			least_error = error;
			C = C_attempt;
			sigma = sigma_attempt;
		end
	endfor
endfor

% =========================================================================

end
