function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================`
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Initial values
	% size(X) == ;
	% size(y) == ;
	% size(theta) == ;

% Sees like this assumes you've included The bias terms X_0 in X. 
% That means fmincg probably adds the bias terms before passing to this cost function.
% Annoying; I'll have to switch between X and Xbias when testing this function alone vs with ex5.

% Xbias = [ones(size(X, 1), 1) X];

% fprintf('sizes\n')
% size(X)
% size(y)
% size(theta)
% size(X * theta)
% size(Xbias * theta)

% J = (sum((Xbias * theta - y).^2) + theta(2:end).^2) / (2*m);
J = (sum((X * theta - y).^2) + lambda * sum(theta(2:end).^2)) / (2*m);

% size(X)
% size(y)
% size(theta)


grad = (X' * (X * theta - y) + lambda * theta) / m;
% grad(1) = X(1) * sum(X * theta - y) / m;
grad(1) = grad(1) - (lambda * theta(1) /  m); % This is cleaner to me for some reason
% grad = (Xbias' * (Xbias * theta - y) + lambda * theta) / m;







% =========================================================================

grad = grad(:);

end
