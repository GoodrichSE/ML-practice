function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%test_theta = zeros(cols(X),1); % X has already had x0 added


% Each row x(ind,:) will need to be multiplied by theta_all (each label's theta)
% Generates a vector of probabilities
% Find the max
% That is p(ind)
% If theta_all is arranged with each label in columns, probs will have results in rows
% Find max of the rows

probs = X * all_theta'; % Probabilites. Columns: labels. Rows: input examples.
% val will be the probability of the most probable response. We just want the index of the most probable.
[val, p] = max(probs, [], 2);

% Alternative:
% [val, ind] = max(probs'); % Max of each of probs's rows. (Returns a row)
% p = ind';


% =========================================================================


end
