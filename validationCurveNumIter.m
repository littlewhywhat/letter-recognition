function [num_iters_vec, error_train, error_val] = validationCurveNumIter(nn_structure, X_train, y_train, X_val, y_val, lambda)

num_iters_vec = [1 3 9 10 30 60 90 110 160 200 240 260 300]';

error_train = zeros(length(num_iters_vec),1);
error_val = zeros(length(num_iters_vec),1);

for i = 1:length(num_iters_vec)
    options = optimset('MaxIter', num_iters_vec(i));
    initial_nn_params = randInitializeNNParams(nn_structure); 
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train, y_train, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); 
    [error_train(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_train, y_train, lambda);
    [error_val(i), dummy]  = nnCostFunction(nn_params, nn_structure, X_val, y_val, lambda);
end

end
