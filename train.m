function train(data_file, nn_structure, lambda, ratios)

%% =========== Part 1: Loading and Visualizing Data =============

close all
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load(data_file);

nn_structure = [size(X,2), nn_structure, max(y)];

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Initializing Parameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_nn_params = randInitializeNNParams(nn_structure);

%% =================== Part 3: Training NN ===================
% divide data on training, validation and testing sets
sel = randperm(size(X,1));
part1_size = size(X,1)/sum(ratios) * ratios(1);
X_train = X(sel(1 : part1_size), :);
y_train = y(sel(1 : part1_size), :);
X_test = X(sel(part1_size + 1 : end), :);
y_test = y(sel(part1_size + 1 : end), :);

% compute 

fprintf('\nSize of training set: %d\n', size(X_train, 1));
for i = 1:length(nn_structure),
    fprintf('\nLayer %d size: %f\n', i, nn_structure(i));
end
fprintf('\nLambda: %f\n', lambda);
fprintf('\nTraining Neural Network... \n\n')

compute = true;

while (compute)

max_iter = input('Input max number of iterations: ');

fprintf('\nMax number of iterations: %f\n', max_iter);

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
    % uniqueness check
    [mat, index] = unique(Theta_i, 'rows', 'first');
    repeatedIndex = setdiff(1:size(Theta_i, 1), index);
    if (size(repeatedIndex, 2) != 0)
        fprintf('\n Collision!\n');
    endif
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

compute = yes_or_no('Continue computation for same parameters? ');

initial_nn_params = nn_params;

end

end
