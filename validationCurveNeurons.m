function [neurons_vec, error_train, error_val] = validationCurveLambda(nn_structure, X_train, y_train, X_val, y_val, lambda)

neurons_vec = [1 3 9 10 20 30 40 50 60 70 80 90 100 110]';

error_train = zeros(length(neurons_vec),1);
error_val = zeros(length(neurons_vec),1);

options = optimset('MaxIter', 50);

for i = 1:length(neurons_vec)
    nn_structure(2) = neurons_vec(i);
    initial_nn_params = randInitializeNNParams(nn_structure); 
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); 
    [error_train(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_train, y_train, lambda);
    [error_val(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_val, y_val, lambda);
end

end
