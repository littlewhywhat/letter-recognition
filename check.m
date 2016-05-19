
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
nn_structure = [input_layer_size, hidden_layer_size, num_labels];

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data/data_20x20_10x500.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ===== Part 2: Loading Pre-initialized Parameters for tests =====

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('data/checkweights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ============ Part 3: Checking Cost (Feedforward) ================

fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, nn_structure, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== Part 4: Checking Cost with Regularization ===========

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, nn_structure, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 5: Sigmoid Gradient Check =============

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===== Part 6: Checking Backpropagation without Regulirization =====

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ====== Part 7: Checking Backpropagation with Regularization =======

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, nn_structure, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 3): %f ' ...
         '\n(this value should be about 0.576051)\n\n'], debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 8: Initializing Parameters For Training =============

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_nn_params = [];
for i = 2:length(nn_structure),
    initial_Thetai = randInitializeWeights(nn_structure(i-1), nn_structure(i));
    initial_nn_params = [initial_nn_params ; initial_Thetai(:)];
end

%% =================== Part 9: Training NN ===================

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);

lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, nn_structure, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% ================= Part 10: Visualize Weights =================

start_i = 0; 
for i = 2:length(nn_structure) - 1, 
    fprintf('\nVisualizing layer %d.\n', i); 
    
    end_i = start_i + nn_structure(i) * (nn_structure(i - 1) + 1); 
    
    Theta_i = reshape(nn_params(start_i + 1: end_i), 
                      nn_structure(i), nn_structure(i - 1) + 1); 
    displayData(Theta_i(:, 2:end)); 
    
    start_i = start_i + (end_i - start_i); 
    
    fprintf('Program paused. Press enter to continue.\n'); 
    pause; 
end 

%% ================= Part 11: Predicting ====================

pred = predict(nn_params, nn_structure, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

clear
