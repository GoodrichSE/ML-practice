function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% a1 will have 401 units (input)
% a2 will have 26 units
% a3 will have 10 units (output)
% Should be able to handle any size though

% Random thought: We could probably do forward propogation with an unknown number of theta's provided. 
% Theta would be an unrolled vector of all theta_i_j_k's. We would also need a matrix of Lx2
% to provide the dimensions of each theta matrix, where L is the number of stages.
% Then we could unroll it and run it through a for-loop L number of times. 
% You could also write a method for unrolling theta given a variable number of theta matrices, 
% complete with initial values. Then you provide theta matrices,and that completely describes your neural net.



%%%%%%%%%%%%%%%%%%%%%%%
%Single Example Method%
%%%%%%%%%%%%%%%%%%%%%%%
% This is done as recommended by the text, by passing a single training example through forward propogation
% and back in a for-loop. Later, I will try to do this with all examples at once, mostly to see if the
% results or computation time is noticibly different.

% This doesn't fully work. I got vectorized implementation working at it was several times faster, so I
% didn't finish this.

%{
Delta2 = Delta1 = 0; % Accrued delta values
reg = (lambda / (2*m)) * sum([Theta1(:);Theta2(:)].^2);

for t = 1:m
	% Forward Propagation
		% X is 400x1
		% Theta1 is 25x401 after adding ones
		% Theta2 is 10x26 after adding ones
	a1 = [1; X(t,:)']; % size(a1) = 401x1
	z2 = Theta1 * a1;
	sigi = sigmoid(z2);
	a2 = [1; sigi];
	%a2 = [1; sigmoid(z2)]; % size(a2) == 26x1
	z3 = Theta2 * a2;
	a3 = sigmoid(z3); % size (a3) == 10x1
	
	y_t = y(t) == vec(1:size(a3,1));
	j = (trace(-y_t' * log(a3) - (1-y_t)' * log(1-a3)) / m);
	
	% Back Propagation
		% Theta1 is 25x400 because we aren't adding ones
		% Theta2 is 10x25, similarly
		% It seems like the back prop testing algorithm expects us to calculate gradients for biases,
		% but then ignore them when accruing Deltas.
	% % a1 = a1(2:end); % It's ok to trim these if we are adding the forward prop bias within the same for-loop
	% % a2 = a2(2:end); % At least, I think so... and it is messy to do this; edits get dangerous.
	
	d3 = a3 - (y(t) == vec(1:size(a3,1)));
	%d2 = (Theta2' * d3) .* sigmoidGradient(z2); % Don't use this because it calls sigmoid() two extra times
	% d2 = (Theta2' * d3) .* (sigi .* (1 - sigi)); % size(d2) == 25x1
	d2 = (Theta2' * d3) .* (a2 .* (1 - a2)); % size(d2) == 26x1
	%d2 = (Theta2' * d3) .* a2 .* (1 - a2); % Can we use this, given that we've trimmed a2?
	%d2 = (Theta2' * d3) .* a2(2:end) .* (1-a2(2:end)); % This is best if trimming is fast
	
	Delta2 = Delta2 + d3 * a2'; % size(Delta2) == 10x25
	Delta1 = Delta1 + d2(2:end) * a1'; % size(Delta1) == 25x400
	
endfor

J = sum(j) + reg;

Theta1_grad = Delta1/m;	
Theta2_grad = Delta2/m;
%}


%%%%%%%%%%%%%%%%%
%Full-Set Method%
%%%%%%%%%%%%%%%%%

% This is an attempt to pass the full 5000-element training set instead of a single training example.
% It will pass the inputs through as a 5000x400 matrix. It will pass once through the forward propogation
% step, ending in a 5000x10 matrix showing the probability that that training example resulted in any
% of the 10 possible outputs. Then it will pass once through back-propogation to get the gradients.

% Forward Propagation
	% X is 5000x400
	% Theta1 is 25x401 after adding ones
	% Theta2 is 10x26 after adding ones
a1 = [ones(size(X, 1), 1), X]; % Add bias term. size(a1) == 5000x401
% Saving the result of sigmoid without bias terms, for use in back prop
z2 = sigmoid(a1 * Theta1'); % size(z2) == 5000x25
a2 = [ones(size(z2, 1), 1) z2]; % Add bias term. size(a2) == 5000x26
% % Not bothering to remove bias terms for back prop
% a2 = sigmoid(a1 * Theta1'); % size(a2) == 5000x25
% a2 = [ones(size(a2, 1), 1) a2]; % Add bias term. size(a2) == 5000x26
a3 = sigmoid(a2 * Theta2'); % size (a3) == 5000x10

	% The hypothesis in this case is kept as a 5000x10 matrix to reflect accuracy as compared to the 
	% classification vectors (e.g. [0 0 0 1 0 0 0 0 0]). An alternative is to find the best fit for each
	% row in the hypothesis to convert it into a vector. Then we could compare that to y (as a vector)
	% That would be something like this:

%[junk, indx] = ind(max(a3, [], 2)); % Find max of each row and the index (along the row) of that value.
%hyp = indx; % The index within the row is the value we want. This creates a 5000-dim vector.
%err = hyp - y;

% Convert y from values to classification vectors
y_vec = y==1:num_labels; % size(y_vec) = 5000x10, 5000 10-class classification vectors

% Regularization
%reg = (lambda / (2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
reg = (lambda / (2*m)) * sum([Theta1(:,2:end)(:);Theta2(:,2:end)(:)].^2);
	% Do not include the bias terms in this! It will mess everything up and be hard to find.
	% Also do not actually trim theta's. Just return trimed values but do not use that to overwrite theta.

% Cost
	% We need element-wise multiplication for the nested sum in the cost function formula for this Sto work.
	% Equations of the form (y' * a3) worked for vector matrices, but in rectangular matrices
	% we get a rectangular matrix with incorrect sums. The correct sums are only along the main diagonal.
	% The trace() function sums along the diagonal, as does sum(<square-matrix> * I).
	% See https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/AzIrrO7wEeaV3gonaJwAFA
%J = (1/m) * (trace(-y' * log(a3)) - trace((1-y)' * log(1-a3)));
J = (trace(-y_vec' * log(a3) - (1-y_vec)' * log(1-a3)) / m) + reg;


% Back Propagation (without pre-trimming)
% Delta2 = Delta1 = 0; % Accrued delta values
% Theta1 = Theta1(:,2:end); % Remove bias terms for back propogation
% Theta2 = Theta2(:,2:end);
% a2 = a2(:,2:end);
% a1 = a1(:,2:end);

	% Apparently, we need to include bias terms for calculating delta's for Theta. Otherwise, the test
	% function for this course will find incorrect matrix dimensions. We only need to omit biases when
	% we do the final accrual of delta terms. I'm not sure yet if this is a best-practice thing or just
	% a consequence of how they've set up their tests. I understood the lectures differently; I will 
	% see for sure in the coming weeks.
	
d3 = a3 - y_vec; % d3 = size(5000x10)
% From https://www.coursera.org/learn/machine-learning/resources/EcbzQ : 
% Use 2 .* (1 - z2) instead of sigmoidGradient(z2)
% Without bias terms
d2 = (d3 * Theta2(:,2:end)) .* z2 .* (1 - z2); % size(d2) == 5000x25
% d2 = [zeros(size(d2, 1), 1) d2]; % This feels like an awful hack. Just trying to get dimensions to match.
% With bias terms
% d2 = (d3 * Theta2) .* a2 .* (1 - a2); % size(d3 * Theta2) == 5000x26, size(d2) == 5000x26
% (It's basically the chain rule applied to find the derivative of the formula we use to find a3.)

% No need to find d1, of course. Can't have errors in calculating a1 since those are our givens.

% Accrue errors for our Theta's
Delta2 = d3' * a2; % size(Delta2) == 10x26
% % Bias terms cannot backpropogate, so we ignore the ones in the output for this stage.
% Delta1 = d2(:,2:end)' * a1; % size(Delta1) == 25x401
% Use this if we've taken care of the bias term in calculating d2 above.
Delta1 = d2' * a1; % size(Delta1) == 25x400
	% I am using this form because I can quickly sum all the delta values with matrix multiplication.
	% I want to sum the terms at any node across all training examples. I should end up with a matrix that holds
	% the accrued error for the Theta's. Thus, the dimensions should also end up the same as the Theta's.
	% Part of the reason this works is because I am not re-using values from one stage to calculate the next.
	% Nor am I using the values from one training example to influence the next. It is all simple sums and
	% multiplication, so I can crunch all the numbers at the same time. That means that not only am I allowed 
	% to use matrices this way, but I can also apply this in a for-loop to express any number of layers.

% % Regularize errors
	% (With a few different methods)

% Delta2_reg = (lambda / m) * Theta2(:,2:end);
% Delta1_reg = (lambda / m) * Theta1(:,2:end);
% Delta2(:, 2:end) = Delta2(:, 2:end) + Delta2_reg;
% Delta1(:, 2:end) = Delta1(:, 2:end) + Delta1_reg;

% Delta2_reg = (lambda / m) * Theta2(:,2:end);
% Delta1_reg = (lambda / m) * Theta1(:,2:end);
% Theta2_grad = Delta2 / m;
% Theta2_grad(:, 2:end) = Delta2(:, 2:end) / m + Delta2_reg;
% Theta1_grad = Delta1 / m;
% Theta1_grad(:, 2:end) = Delta1(:, 2:end) / m + Delta1_reg;
	% Inefficient and somehow made bad dimensions
	
% Theta1_grad = (Delta1(:, 2:end) + lambda * Theta1(:, 2:end)) / m;
% Theta1_grad = [Delta1(:,1) Theta1_grad];
% Theta2_grad = (Delta2(:, 2:end) + lambda * Theta2(:, 2:end)) / m;
% Theta2_grad = [Delta2(:,1) Theta2_grad];
	% More efficient, but also did not work for some reason
	
Delta1_reg = lambda * Theta1(:,2:end);
Delta2_reg = lambda * Theta2(:,2:end);
Delta1_reg = [zeros(size(Delta1_reg, 1), 1) Delta1_reg];
Delta2_reg = [zeros(size(Delta2_reg, 1), 1) Delta2_reg];
Theta1_grad = (Delta1 + Delta1_reg) / m;
Theta2_grad = (Delta2 + Delta2_reg) / m;
	% This seems to work best.

% Theta1(:,1) = 0;
% Theta2(:,1) = 0;
% Theta1_grad = (Delta1 + (lambda * Theta1)) / m;
% Theta2_grad = (Delta2 + (lambda * Theta2)) / m;
	% As suggested by https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q
	% Also works.
	
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
