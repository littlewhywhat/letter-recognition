function analyze(data_file, nn_structure, lambda, ratios)
    
    close all;
    load(data_file);
    nn_structure = [size(X,2), nn_structure, max(y)];

%% =========== Separation of Data =============
    % divide data on training, validation and testing sets
    sel = randperm(size(X,1));
    part1_size = size(X,1)/sum(ratios) * ratios(1);
    part2_size = size(X,1)/sum(ratios) * ratios(2);
    X_train = X(sel(1 : part1_size), :);
    y_train = y(sel(1 : part1_size), :);
    X_val = X(sel(part1_size + 1 : part1_size + part2_size), :);
    y_val = y(sel(part1_size + 1 : part1_size + part2_size), :);
    X_test = X(sel(part1_size + part2_size + 1 : end), :);
    y_test = y(sel(part1_size + part2_size + 1 : end), :);

%% =========== Learning Curve =============
 
    fprintf('Computing learning curve...\n');

    % compute and  display learning curve
    [error_train, error_val] = learningCurve(nn_structure, X_train, y_train, X_test, y_test, lambda);
    
    m = size(error_train, 1);
    
    figure(1);
    plot(1:m, error_train, 1:m, error_val);

    title(sprintf('Learning Curve (lambda = %f)', lambda));
    xlabel('Number of training examples')
    ylabel('Error')
    axis([0 7 0 4])
    legend('Train', 'Cross Validation')

    fprintf('Program paused. Press enter to continue.\n');
    pause;

%% =========== Validation for Selecting Lambda =============

    fprintf('Computing validation for different lambdas...\n');

[lambda_vec, error_train, error_val] = ...
    validationCurveLambda(nn_structure, X_train, y_train, X_val, y_val);

    close all;
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('lambda');
    ylabel('Error');

    fprintf('lambda\t\tTrain Error\tValidation Error\n');
    for i = 1:length(lambda_vec)
    fprintf(' %f\t%f\t%f\n', ...
        lambda_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;

% ============ Train for special lambda and predict on test set =========== %%
  
   try_lambda = true;
    while (try_lambda) 
        lambda = input('Choose lambda: ');
        options = optimset('MaxIter', 50);
        initial_nn_params = randInitializeNNParams(nn_structure);
        costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        pred = predict(nn_params, nn_structure, X_test);
        fprintf('Testing Set Accuracy: %f\n\n\n', mean(double(pred == y_test)) * 100);
        try_lambda = yes_or_no('Try another lambda?');
    end

%% =========== Validation for Selecting Number of hidden neurons =============

    fprintf('Computing validation for different number of hidden neurons...\n');

[neurons_vec, error_train, error_val] = ...
    validationCurveNeurons(nn_structure, X_train, y_train, X_val, y_val, lambda);

    close all;
    plot(neurons_vec, error_train, neurons_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('Number of neurons');
    ylabel('Error');

    fprintf('Neurons\t\tTrain Error\tValidation Error\n');
    for i = 1:length(neurons_vec)
    fprintf(' %f\t%f\t%f\n', ...
        neurons_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;

% ============ Train for special lambda and predict on test set =========== %%

    try_neurons = true;
    while (try_neurons) 
        neurons = input('Choose number of hidden neurons: ');
        nn_structure(1) = neurons;
        options = optimset('MaxIter', 50);
        initial_nn_params = randInitializeNNParams(nn_structure);
        costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        pred = predict(nn_params, nn_structure, X_test);
        fprintf('Testing Set Accuracy: %f\n\n\n', mean(double(pred == y_test)) * 100);
        try_neurons = yes_or_no('Try another number of hidden neurons?');
    end

%% =========== Validation for Selecting Number of iterations =============

    fprintf('Computing validation for different number of iterations...\n');

[num_iters_vec, error_train, error_val] = ...
    validationCurveNumIter(nn_structure, X_train, y_train, X_val, y_val, lambda);

    close all;
    plot(num_iters_vec, error_train, num_iters_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('Number of iterations');
    ylabel('Error');

    fprintf('Number of iterations\t\tTrain Error\tValidation Error\n');
    for i = 1:length(num_iters_vec)
    fprintf(' %f\t%f\t%f\n', ...
        num_iters_vec(i), error_train(i), error_val(i));
    end

    fprintf('Program paused. Press enter to continue.\n');
    pause;

% ============ Train for special number of iterations and predict on test set =========== %%
    try_num_iters = true;
    while (try_num_iters) 
        num_iters = input('Choose number of hidden num_iters: ');
        options = optimset('MaxIter', num_iters);
        initial_nn_params = randInitializeNNParams(nn_structure);
        costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
        pred = predict(nn_params, nn_structure, X_test);
        fprintf('Testing Set Accuracy: %f\n\n\n', mean(double(pred == y_test)) * 100);
        try_neurons = yes_or_no('Try another number of hidden neurons?');
    end
end
