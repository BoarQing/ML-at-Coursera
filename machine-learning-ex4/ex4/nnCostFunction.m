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

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%X's size:5000 400
%y's size:5000 1
%Theta1's size:25 401
%Theta2's size:10 26
a1 = [ones(m, 1) X];
z2 = a1 * Theta1'; %5000 * 25
a2 = sigmoid(z2);  
a2 = [ones(m, 1) a2]; %5000 * 26
z3 = a2 * Theta2';
a3 = sigmoid(z3); %5000 * 10
K = size(Theta2, 1);
for i = 1 : K,
    expectedY = y == i;
    predictedY = a3(:, i);
    J = J + sum(-expectedY .* log(predictedY) - (1 - expectedY) .* log(1 - predictedY));
end
J = J / m;

Theta1Col = size(Theta1, 1);
Theta2Col = size(Theta2, 1);
regularizationEach = sum(sum(Theta1 .^ 2)) + sum(sum(Theta2 .^ 2)) - sum(sum(Theta1(:, 1) .^ 2)) - sum(sum(Theta2(:, 1) .^ 2));
regularization = regularizationEach * lambda / 2 / m;
J = J + regularization;

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
delta3 = [];
for i = 1 : K,
    expectedY = y == i;
    predictedY = a3(:, i);
    err = predictedY - expectedY;
    delta3 = [delta3, err];
end

%why it did not need .* sigmoidGradient(z3)?
%Because dJ/dTheta = (predictY - expectedY) .* a2 from ex3's pdf!
%So the delta3 is already include sigmoidGradient(z3);
%delta3 is dE/dy * dy/dx alrady

Theta2_grad = delta3' * a2 ./ m; 
deltaE_deltaA = (delta3 * Theta2)';
delta2 =  deltaE_deltaA(2 : end, :) .* sigmoidGradient(z2)';
Theta1_grad = delta2 * a1 ./ m; 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% =========================================================================
Theta1_Regularization = Theta1 * lambda / m; %it is okay to add weight to w0
Theta1_Regularization(:, 1) = 0;
Theta2_Regularization = Theta2 * lambda / m;
Theta2_Regularization(:, 1) = 0;
Theta1_grad = Theta1_grad + Theta1_Regularization;
Theta2_grad = Theta2_grad + Theta2_Regularization;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
