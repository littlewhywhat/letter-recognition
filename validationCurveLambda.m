function [lambda_vec, error_train, error_val] = validationCurveLambda(nn_structure, X_train, y_train, X_val, y_val, max_iters)

lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 ]';

error_train = zeros(length(lambda_vec),1);
error_val = zeros(length(lambda_vec),1);

options = optimset('MaxIter', max_iters);

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    initial_nn_params = randInitializeNNParams(nn_structure); 
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); 
    [error_train(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_train, y_train, lambda);
    [error_val(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_val, y_val, lambda);
end

end
