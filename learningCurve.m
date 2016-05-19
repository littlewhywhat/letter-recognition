function [error_train, error_val] = learningCurve(nn_structure, X_train, y_train, X_val, y_val, lambda)

m = min([size(X_train,1), size(X_val, 1)]);

error_size = int32(m / 100);

error_train = zeros(error_size,1);
error_val = zeros(error_size,1);

options = optimset('MaxIter', 50);

i = 100;
for j = 1:error_size - 1
    initial_nn_params = randInitializeNNParams(nn_structure); 
    costFunction = @(p) nnCostFunction(p, nn_structure, X_train(1:i, :), y_train(1:i,:), lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); 
    [error_train(j), dummy]  = nnCostFunction(nn_params, nn_structure, X_train(1:i, :), y_train(1:i, :), 0);
    [error_val(j), dummy]  = nnCostFunction(nn_params, nn_structure, X_val, y_val, 0);
    i += 100;
end

end
