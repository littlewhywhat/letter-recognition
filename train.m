function train(data_file, nn_structure, max_iter, lambda, ratio1, ratio2)

%% =========== Part 1: Loading and Visualizing Data =============

close all
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load(data_file);

%removeRedund(X);

nn_structure = [size(X,2), nn_structure];

% input_layer_size = size(X, 2);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_nn_params = [];
for i = 2:length(nn_structure),
    initial_Thetai = randInitializeWeights(nn_structure(i-1), nn_structure(i));
    initial_nn_params = [initial_nn_params ; initial_Thetai(:)];
end

%% =================== Part 3: Training NN ===================

% divide data on training and testing sets
[X_train, X_test] = divideData(X, ratio1, ratio2, nn_structure(end));
[y_train, y_test] = divideData(y, ratio1, ratio2, nn_structure(end));

fprintf('\nMax number of iterations: %f\n', max_iter);
for i = 1:length(nn_structure),
    fprintf('\nLayer %d size: %f\n', i, nn_structure(i));
end
fprintf('\nLambda: %f\n', lambda);
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', max_iter);
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   nn_structure, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 4: Visualize Weights =================

start_i = 0;
for i = 2:length(nn_structure) - 1,
    fprintf('\nVisualizing layer %d.\n', i);
    end_i = start_i + nn_structure(i) * (nn_structure(i - 1) + 1);
    Theta_i = reshape(nn_params(start_i + 1: end_i),
                      nn_structure(i), nn_structure(i - 1) + 1);
    displayData(Theta_i(:, 2:end));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    start_i = start_i + (end_i - start_i);
end

%% ================= Part 5: Predict =================

pred = predict(nn_params, nn_structure, X_train);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);

pred = predict(nn_params, nn_structure, X_test);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

end
