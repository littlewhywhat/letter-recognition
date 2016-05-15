function [J grad] = nnCostFunction(nn_params, ...
                                   nn_structure,
                                   X, y, lambda)

% Setup some useful variables
m = size(X, 1);
         
% convert y to binary matrix for logistic regression
y_bin = zeros(size(y, 1), nn_structure(length(nn_structure)));
for i = 1:length(y)
    y_bin(i, y(i)) = 1;
end

% perform feedforward computation
start_i = 0;
layers = X(:);
layer_i = X;
reg = 0;
for i = 2:length(nn_structure),
  % selection indices
  end_i = start_i + nn_structure(i) * (nn_structure(i-1) + 1);
  % extract neccessary theta
  Thetai = reshape(nn_params(start_i + 1 : end_i), 
                   nn_structure(i), (nn_structure(i-1) + 1));
  % include bias terms - examples x (nn_structure(i-1) + 1) 
  layer_i = [ones(size(layer_i, 1), 1), layer_i]; 
  % examples x nn_structure(i)
  layer_i = layer_i * Thetai';
  % save computed layer for backpropagation
  layers = [layer_i(:); layers];
  % compute final layer_i for next iteration
  layer_i = sigmoid(layer_i); 
  % selection indices
  start_i = start_i + (end_i - start_i);
  % compute values for regulirization
  Thetai(:, 1) = zeros(size(Thetai, 1), 1);
  reg = reg + sum(sum(Thetai .^ 2, 2));
end

hx = layer_i;

% compute cost function
sumk = sum((-y_bin) .* log(hx) - (1 - y_bin) .* log(1 - hx), 2);

J = (1/m) * sum( ... 5000 x 1
                   sumk
               );

% compute regulirization term
reg = (lambda/(2*m)) * reg;

J = J + reg;

% perform backpropogation of errors and compute gradient

grad = [];
erri = hx - y_bin;

nn_structure = fliplr(nn_structure);

end_i = numel(nn_params);
start_l = nn_structure(1) * m;
for i = 2:length(nn_structure) - 1,
    % selection indices
    start_i = end_i - (nn_structure(i) + 1) * nn_structure(i-1);
    end_l = start_l + nn_structure(i) * m;
    % extract neccessary theta and layer values used in feedforward computation
    Thetai = reshape(nn_params(start_i + 1 : end_i), 
                   nn_structure(i-1), nn_structure(i) + 1);
    layer_i = reshape(layers(start_l + 1:end_l),
                   m, nn_structure(i));
    % compute gradient
    Thetai_grad = (1/m) * (erri' * [ones(size(layer_i, 1), 1), sigmoid(layer_i)]);
    tmp_Thetai = Thetai;
    tmp_Thetai(:,1) = zeros(size(tmp_Thetai, 1), 1);
    Thetai_grad = Thetai_grad + (lambda/m) * (tmp_Thetai);
    
    grad = [Thetai_grad(:); grad];
    
    % compute error for next layer
    erri = (erri * Thetai);
    erri = erri(:, 2:end);
    erri = erri .* sigmoidGradient(layer_i);
    
    % selection indices
    end_i = end_i - (end_i - start_i);
    start_l = start_l + (end_l - start_l);
end

i = length(nn_structure);
start_i = end_i - (nn_structure(i) + 1) * nn_structure(i-1);
Thetai = reshape(nn_params(start_i + 1 : end_i), 
                 nn_structure(i-1), nn_structure(i) + 1);

Thetai_grad = (1/m) * (erri' * [ones(size(X, 1), 1), X]);
tmp_Thetai = Thetai;
tmp_Thetai(:,1) = zeros(size(tmp_Thetai, 1), 1);
Thetai_grad = Thetai_grad + (lambda/m) * (tmp_Thetai);

grad = [Thetai_grad(:); grad];
% -------------------------------------------------------------

% =========================================================================

end
