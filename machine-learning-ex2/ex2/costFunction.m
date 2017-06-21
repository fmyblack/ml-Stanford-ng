function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for iter = 1 : m
	Xi = X(iter, :);
	co = Xi * theta;
	h = 1 / (1 + e ^ (-co));
	yi = y(iter);
	Ji = ( -1 * yi * log(h) - (1 - yi) * log(1 - h));
	J = J + Ji;
	grad = grad + ((h - yi) * Xi)';
end

J = J / m;
grad = grad / m;


% =============================================================

end
