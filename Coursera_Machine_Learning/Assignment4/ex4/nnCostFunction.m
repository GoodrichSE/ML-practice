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

% a1 will have 400 units (input)
% a2 will have 25 units
% a3 will have 10 units (output)
% Should be able to handle any size though

% We could probably do forward propogation with an unknown number of theta's provided. 
% Theta would be an unrolled vector of all theta_i_j_k's. We would also need a matrix of Lx2
% to provide the dimensions of each theta matrix. Then we could unroll it and run it through
% a for-loop L number of times. You could also write a method for unrolling theta given a 
% variable number of theta matrices, complete with initial values. Then you provide theta matrices,
% and that completely describes your neural net. Anyway...



%%%%%%%%%%%%%%%%%%%%%%%
%Single Example Method%
%%%%%%%%%%%%%%%%%%%%%%%

% This is done as recommended by the text, by passing a single training example through forward propogation
% and back in a for-loop. Later, I will try to do this with all examples at once, mostly to see if the
% results or computation time is noticibly different.

Delta2 = Delta1 = 0; % Accrued delta values

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
	
	
	% Back Propagation
		% Theta1 is 25x400 because we aren't adding ones
		% Theta2 is 10x25, similarly
	a1 = a1(2:end); % It's ok to trim these if we are adding the bias within the same for-loop
	a2 = a2(2:end);
	
	d3 = a3 - (y(t) == vec(1:size(a3,1)));
	%d2 = (Theta2' * d3) .* sigmoidGradient(z2); % Don't use this because it calls sigmoid() two extra times
	d2 = (Theta2(:,2:end)' * d3) .* (sigi .* (1 - sigi));
	%d2 = (Theta2' * d3) .* a2 .* (1 - a2); % Can we use this, given that we've trimmed a2?
	% size(d2) == 25x1
	
	Delta2 = Delta2 + d3 * a2'; % size(Delta2) == 10x25
	Delta1 = Delta1 + d2 * a1'; % size(Delta1) == 25x400
	
endfor

Theta1_grad = Delta1/m;	
Theta2_grad = Delta2/m;

%%%%%%%%%%%%%%%%%
%Full-Set Method%
%%%%%%%%%%%%%%%%%

% This is an attempt to pass the full 5000-element training set instead of a single training example.
% It will pass the inputs through as a 5000x400 matrix. It will pass once through the forward propogation
% step, ending in a 5000x10 matrix showing the probability that that training example resulted in any
% of the 10 possible outputs. Then it will pass once through  back-propogation to get the gradients.
%{
% Forward Propagation
	% X is 5000x400 (5000x401 after adding x_0)
	% Theta1 is 25x401 after adding ones
	% Theta2 is 10x26 after adding ones
a1 = X;
a1 = [ones(size(a1, 1), 1) a1]; % size(a1) == 5000x401, size(Theta1) == 25x401
	%fprintf('stage one:\n%f\n%f\n', size(a1), size(Theta1));
a2 = sigmoid(a1 * Theta1');
a2 = [ones(size(a2, 1), 1) a2]; % size(a2) == 5000x26, size(Theta2) == 26x10
	%fprintf('stage two:\n%f\n%f\n', size(a2), size(Theta2));
a3 = sigmoid(a2 * Theta2'); % size (a3) == 5000x10
	%fprintf('final stage:\n%f\n', size(a3));
hyp = a3; % size(hyp) is 5000x10

% The hypothesis in this case is kept as a 5000x10 matrix to reflect accuracy as compared to the 
% classification vectors (e.g. [0 0 0 1 0 0 0 0 0]). An alternative is to find the best fit for each
% row in the hypothesis and convert it into a vector. That would be something like this:

%[junk, indx] = ind(max(hyp, [], 2)); % Find max of each row and the index (along the row) of that value.
%hyp = indx; % The index within the row is the value we want.

% We need element-wise multiplication for the nested sum in this cost function to work.
% Equations of the form (y' * hyp) worked for vector matrices, but in rectangular matrices
% we get a rectangular matrix with incorrect sums. The correct sums are only along the main diagonal.
% The trace() function sums along the diagonal, as does sum(<square-matrix> * I).
% See https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/AzIrrO7wEeaV3gonaJwAFA

% We can vectorize this if y is a 5000x10 matrix, with rows representing training examples
% and columns representing outcome classes. Each row in y will be a vector of 0's with a single 1 
% in the appropriate position. i.e. y_i = 4 -> [0 0 0 1 0 0 0 0 0 0]. Then we can do (y' * hyp) * I
% (or trace or whatever) for the nested sum. I will be size 10, y' * hyp will be 10x10

% Convert y from values to classification vectors
y_vec = y==1:num_labels; % size(y_vec) = 5000x10, 5000 10-class classification vectors

% Regularization
%reg = (lambda / (2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
reg = (lambda / (2*m)) * sum([Theta1(:);Theta2(:)].^2);

% Cost
%J = (1/m) * (trace(-y' * log(hyp)) - trace((1-y)' * log(1-hyp)));
J = (trace(-y_vec' * log(hyp) - (1-y_vec)' * log(1-hyp)) / m) + reg;


% Back Propagation
	% Theta1 is 25x400 because we aren't adding ones
	% Theta2 is 10x25, similarly
Delta2 = Delta1 = 0; % Accrued delta values
Theta1 = Theta1(:,2:end); % Remove bias terms for back propogation
Theta2 = Theta2(:,2:end);
a2 = a2(:,2:end);
a1 = a1(:,2:end);
	
d3 = hyp - y_vec; % d3 = size(5000x10)

% From https://www.coursera.org/learn/machine-learning/resources/EcbzQ :
d2 = (d3 * Theta2) .* a2 .* (1 - a2); % size(d3 * Theta2) == 5000x25, size(d2) == 5000x25
% (It's basically the chain rule applied to find the derivative of the formula we use to find a3.)

% No need to find d1, of course. Can't have errors in calculating a1 since those are our givens.

% VERIFY: Compare to single training example for guidance
% Accrue errors for our Theta's
Delta2 = d3'*a2; % size(Delta2) == 10x25
Delta1 = d2'*a1; % size(Delta1) == 25x400
% I am using this form because I can quickly sum all the delta values with matrix multiplication.
% I want to sum the terms along the 5000-dimension row. I should end up with a matrix that holds the
% accrued error for the Theta's. Thus, the dimensions should also end up the same as the Theta's.
% Part of the reason this works is because I am not re-using values from one stage to calculate the next.
% Nor am I using the values from one training example to influence the next. It is all simple sums and
% multiplication. That means that not only am I allowed to use matrices this way, but I can also
% apply this in a for-loop to express any number of layers.


%TODO: Theta1 and Theta2 had the bias terms infused into them. Remove the bias from back propogation.
%TODO: is hyp necessary?
%TODO: Test pre-trimed a2 instead of z2 for sigmoid gradient in back prop
%TODO: Don't use bias units in sigmoid functions
%}

Theta1_grad = (Delta1 + lambda * Theta1(:,2:end)) / m;
Theta2_grad = (Delta2 + lambda * Theta2(:,2:end)) / m;
fprintf("Size check:\n")
size(Theta1_grad)
size(Theta2_grad)
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
