function J = cost (X,y,theta);

% X is design matrix
% y is labels (training example targets)

m = size(X,1);	%number of examples
%should verify size of theta as well
hypothesis = X*theta;	%hypothesis predictions for m
e = (hypothesis-y).^2	%errors, squared

J = 1/(2*m) * sum(e);