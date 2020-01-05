function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	%%
	% Debugging matrix multiplication formula
	%%
		
	% fprintf('THETA_0\n')
	% fprintf('summation: \n')
	% s = sum(X * theta - y);
	% S = length(s)
	% fprintf('gradation: \n')
	% s = (alpha / m) * s;
	% S = length(s)
	
	% fprintf('unmodified sum: \n')
	% s = (1 / m) * sum(X * theta - y)
	% fprintf('summed before multiplying by x: \n')
	% s1 = (1 / m) * sum(X * theta - y);
	% s1 = sum(s1 * X(2))
	% fprintf('summed after multiplying by x: \n')
	% s2 = (1 / m) * sum((X * theta - y) * X(2))
	
	
	%%
	%	Formulae
	%%	
	
	storage = [0; 0];	% temporary storage so we can update theta simultaneously

	% Using sum
	% storage(1) = theta(1) - (alpha / m) * sum(X * theta - y);
	% storage(2) = theta(2) - (alpha / m) * sum((X * theta - y) * X(:,2));
	
	% Using transpose vectors to sum
	% storage(1) = theta(1) - (alpha / m) * (X * theta - y)' * ones(m,1)
	storage(1) = theta(1) - (alpha / m) * (X * theta - y)' * X(:,1);
	storage(2) = theta(2) - (alpha / m) * (X * theta - y)' * X(:,2);
		
	% Full vectorization (didnt work)
	% theta = theta - (alpha / m) * (X * theta - y)' * X
	% % theta = theta - (alpha / m) * (X * theta - y) .* X'
	
	% fprintf('Storing into theta.\n')
	theta = storage;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
