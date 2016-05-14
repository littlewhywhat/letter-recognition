function p = predictg(nn_params, nn_structure, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h_i = X;
start_i = 0;
for i = 2:length(nn_structure),
    end_i = start_i + nn_structure(i) * (nn_structure(i - 1) + 1);
    Theta_i = reshape(nn_params(start_i + 1: end_i), 
                        nn_structure(i), nn_structure(i - 1) + 1); 
    h_i = sigmoid([ones(m, 1) h_i] * Theta_i');
    start_i = start_i + (end_i - start_i);
end

% h1 = sigmoid([ones(m, 1) X] * Theta1');

[dummy, p] = max(h_i, [], 2);

% =========================================================================


end
