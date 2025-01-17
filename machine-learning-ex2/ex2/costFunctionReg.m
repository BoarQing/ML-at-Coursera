function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
thetaX = X * theta;
hTheta  = sigmoid(thetaX);
err =  sum(-y .* log(hTheta) - (1 - y) .* log(1 - hTheta)) ./ m;
regular = lambda / m / 2 * (sum(theta .^ 2) - theta(1, 1) ^ 2);
J = err + regular;

grad(1, 1) = sum((hTheta - y) .* X(: , 1)) ./ m;
for gradIndex = 2: length(theta),
    grad(gradIndex, 1) = sum((hTheta - y) .* X(: , gradIndex)) ./ m + lambda / m * theta(gradIndex, 1);
end





% =============================================================

end
