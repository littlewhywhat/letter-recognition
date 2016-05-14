function X = loadLetters(filename, dim)
%{
    Loading and processing test and train data.
%}
% load test_set.bmp into X 
X = imread(filename);
% create check
check = X((1:dim), (1:dim));
% append zeros to right and bottom
X = [X, zeros(size(X, 1), 1)];
X = [X; zeros(1, size(X, 2))];
% unenroll to one big row of letters
X = placeInHorizBlocks(X, dim + 1);
X(end, :) = [];
% reshape to one big column of letters
X = placeInVertBlocks(X, dim + 1);
X(:, end) = [];
% reshape to one big row again
X = placeInHorizBlocks(X, dim);
% reshape to the desired form for neural network 
X = reshape(X(:), dim * dim, size(X,1)/dim * size(X,2)/dim)';
% perform quick check
fprintf('Next output should be equal to: %d \n', dim  * dim );
sum(sum(reshape(X(1, :), dim, dim) == check))
end
