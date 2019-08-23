%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data

load('ex4data1.mat');
m = size(X, 1);



%% ================ Part 2: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

%fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];


lambda = 0;
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


X = [ones(m,1) X];
a2 = sigmoid(X*Theta1'); 
%size(a2)
a2 = [ones(m,1) a2];
%size(a2)
a3 = sigmoid(a2*Theta2');
%size(a3)
K = size(a3,2);
y_new = zeros(m,K);
for k=1:K
y_new(:,k)	= y == k;
end
	
J = -1/m * sum(sum(y_new.*log(a3) + (1-y_new).*log(1 - a3))) ...
+ lambda/(2*m)*(sum(Theta1(:,2:end)(:).^2) + sum(Theta2(:,2:end)(:).^2));


%for i=1:m
%	a1 = X(i,:) ;
%	a2 = sigmoid(a1*Theta1'); 
%	a2 = [1 a2];
%	a3 = sigmoid(a2*Theta2'); 
%	delta3 = a3 - y_new(i,:);
%	z2 = [1 sigmoid(a1*Theta1')];
%	delta2 = delta3 * Theta2 .*sigmoidGradient(z2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Theta1_grad = Theta1_grad + delta2(2:end)'*a1 ;
%	Theta2_grad = Theta2_grad + delta3'*a2 ;
%Theta1_grad = Theta1_grad/m;
%Theta2_grad = Theta2_grad/m;


delta3 = a3 - y_new;
a1=X;
delta2 = delta3 * Theta2(:,2:end) .*sigmoidGradient(a1*Theta1');
Theta1_grad = 1/m * (Theta1_grad + delta2'*a1 );
Theta2_grad = 1/m* (Theta2_grad + delta3'*a2) ;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

