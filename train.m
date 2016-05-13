function train(data_file, hidden_layer_size, num_labels, max_iter, lambda, ratio1, ratio2)

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load(data_file);

removeRedund(X);

input_layer_size = size(X, 2);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 2: Initializing Parameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 3: Training NN ===================

% divide data on training and testing sets
[X_train, X_test] = divideByRow(X, ratio1, ratio2, num_labels);
[y_train, y_test] = divideByRow(y, ratio1, ratio2, num_labels);

fprintf('\nMax number of iterations: %f\n', max_iter);
fprintf('\nNumber of labels: %f\n', num_labels);
fprintf('\nHidden layer size: %f\n', hidden_layer_size);
fprintf('\nInput layer size: %f\n', input_layer_size);
fprintf('\nLambda: %f\n', lambda);
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', max_iter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Visualize Weights =================

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 5: Predict =================

pred = predict(Theta1, Theta2, X_train);

%disp([pred, y_train]);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

%displayData(X_test(1:10, :));

pred = predict(Theta1, Theta2, X_test);

%disp([pred, y_test(1:10, :)]);


fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

end
