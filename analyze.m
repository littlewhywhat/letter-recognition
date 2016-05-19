function analyze(data_file, nn_structure, ratios)
    
%% close all previous plots (figures)
    close all;

%% ================ Prepare data ====================
    % load data
    load(data_file);
    % visualize data
    sel = randperm(size(X, 1));
    sel = sel(1:100);

    displayData(X(sel, :));

    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
    % set initial parameters
    nn_structure = [size(X,2), nn_structure, max(y)];
    max_iters = 50;
    lambda = 1;
    initial_nn_params = randInitializeNNParams(nn_structure);

%% ============= Separation of Data =================
%% Divide data on training, validation and testing sets 
%% using ratios

    sel = randperm(size(X,1));
    part1_size = size(X,1)/sum(ratios) * ratios(1);
    part2_size = size(X,1)/sum(ratios) * ratios(2);
    X_train = X(sel(1 : part1_size), :);
    y_train = y(sel(1 : part1_size), :);
    X_test = X(sel(part1_size + 1 : part1_size + part2_size), :);
    y_test = y(sel(part1_size + 1 : part1_size + part2_size), :);
    X_eval = X(sel(part1_size + part2_size + 1 : end), :);
    y_eval = y(sel(part1_size + part2_size + 1 : end), :);
    
%% ============== Result for initial paramaters =====
    fprintf('Computing result for initial parameters...\n');
    options = optimset('MaxIter', max_iters);
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    pred = predict(nn_params, nn_structure, X_eval);
    fprintf('Evaluation Set Accuracy: %f', mean(double(pred == y_eval)) * 100);

%% ============== Visualize network parameters ======

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


%% ============== Learning Curve ====================

    fprintf('Computing learning curve...\n');

[error_train, error_val] = ...
    learningCurve(nn_structure, X_train, y_train, X_test, y_test, lambda, max_iters);
        
    % display the learning curve on new figure 
    figure;
    num_of_examples_axis = (1:size(error_train, 1)) * 100;
    plot(num_of_examples_axis, error_train, num_of_examples_axis, error_val);

    title(sprintf('Learning Curve (lambda = %f)', lambda));
    xlabel('Number of training examples')
    ylabel('Error')
    axis([num_of_examples_axis(1) max(num_of_examples_axis) 0 max([error_train; error_val]) + 0.5])
    legend('Train set', 'Test set')

    fprintf('Program paused. Press enter to continue.\n');
    pause;

%% =========== Choice of number of hidden neurons =============

    fprintf('Computing errors for train and test sets for different number of hidden neurons...\n');

[neurons_vec, error_train, error_val] = ...
    validationCurveNeurons(nn_structure, X_train, y_train, X_test, y_test, lambda, max_iters);

    % display the change of error depending on number of neurons on new figure
    figure;
    plot(neurons_vec, error_train, neurons_vec, error_val);
    legend('Train set', 'Test set');
    xlabel('Number of neurons');
    ylabel('Error');

    % print the values in three columns     
    fprintf('Neurons\t\tTrain Error\tTest Error\n');
    for i = 1:length(neurons_vec)
    fprintf(' %f\t%f\t%f\n', ...
        neurons_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;

% ============ Train for special number of neurons and predict on evaluation set =========== %%

    neurons = input('Choose final number of hidden neurons: ');
    nn_structure(2) = neurons;
    % need to update initial paramaters to match number of neurons
    initial_nn_params = randInitializeNNParams(nn_structure);
    options = optimset('MaxIter', max_iters);
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    pred = predict(nn_params, nn_structure, X_eval);
    fprintf('Evaluation Set Accuracy: %f\n\n\n', mean(double(pred == y_eval)) * 100);

%% =========== Choice of Lambda =============

    fprintf('Computing errors for train and test sets for different lambdas...\n');

[lambda_vec, error_train, error_val] = ...
    validationCurveLambda(nn_structure, X_train, y_train, X_test, y_test, max_iters);

    % display the change of error depending on lambda on new figure
    figure;
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Train set', 'Test set');
    xlabel('Lambda');
    ylabel('Error');

    % print the values in three columns
    fprintf('lambda\t\tTrain Error\tTest Error\n');
    for i = 1:length(lambda_vec)
    fprintf(' %f\t%f\t%f\n', ...
        lambda_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;

%% ============ Train for special lambda and predict on evaluation set =========== %%
  
    lambda = input('Choose final lambda: ');
    options = optimset('MaxIter', max_iters);
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    pred = predict(nn_params, nn_structure, X_eval);
    fprintf('Evaluation Set Accuracy: %f\n\n\n', mean(double(pred == y_eval)) * 100);
    
%% =========== Choice of number of iterations =============

    fprintf('Computing errors for train and test sets for different number of iterations...\n\n');

[num_iters_vec, error_train, error_val] = ...
    validationCurveNumIter(nn_structure, X_train, y_train, X_test, y_test, lambda);

    % display the change of error depending on number of iterations on new figure
    figure;
    plot(num_iters_vec, error_train, num_iters_vec, error_val);
    legend('Train set', 'Test set');
    xlabel('Number of iterations');
    ylabel('Error');

    % print the values in three columns     
    fprintf('Iterations\tTrain Error\tTest Error\n');
    for i = 1:length(num_iters_vec)
    fprintf(' %f\t%f\t%f\n', ...
        num_iters_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;
   
%% ============== Result for final paramaters =====
    max_iters = input('Choose final number of iterations: ');
    
    fprintf('Computing result for final parameters...\n');
    options = optimset('MaxIter', max_iters);
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    pred = predict(nn_params, nn_structure, X_eval);
    fprintf('Evaluation Set Accuracy: %f', mean(double(pred == y_eval)) * 100);

%% ============== Visualize network parameters ======

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


end
