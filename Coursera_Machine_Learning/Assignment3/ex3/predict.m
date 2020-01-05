function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Append 1's to first row
X = [ones(m,1) X];
% X now has dim m x 401
% m is 1; this works on a single sample
%fprintf('X dim: %f\n', size(X))

%fprintf('theta1 dim: %f\n', size(Theta1))
%fprintf('theta2 dim: %f\n', size(Theta2))

% First layer
% Theta 1 has dim 25x401
z1 = sigmoid(X * Theta1');
%fprintf('z1 dim: %f\n', size(z1))

% Append 1's to beginning
z1 = [ones(size(z1,1),1) z1];
%fprintf('altered z1 dim: %f\n', size(z1))

% Second layer
% Theta 2 has dim 10x26
z2 = sigmoid(z1 * Theta2');
%fprintf('z2 dim: %f\n', size(z2))

% Loop

% 

% Generate probabilities, per sample (cols) and per possible output (rows)
z2;
% Of all probabilities, choose the largest
[val, ind] = max(z2, [], 2);
%fprintf('val, ind: %f %f\n', val, ind)

p = ind;
%fprintf('p: %f\n', p)
% =========================================================================


end
