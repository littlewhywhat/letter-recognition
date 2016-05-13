function X = loadLetters(filename, dim)
%{
    Loading and processing test and train data.
%}
% load test_set.bmp into X 1195x919 (26*45+25 X 20*45+19)
X = imread(filename);
% append zeros to right and bottom (26*46 x 20*46)
X = appendToRightColumn(X, 0);
X = appendToLowRow(X, 0);
% create check
check = X(1:dim, 1:dim);
% unenroll to 46x26*20*46 (one big row of letters)
X = resh(X, dim);
% cut low row
X = removeLowRow(X);
% reshape to 26*20*46x46 (one big column of letters)
X = resh2(X, dim);
% cut right column
X = removeRightColumn(X);
% now we are in 45
dim = dim - 1;
% reshape to one big row again
X = resh(X, dim);
% reshape to the desired form for neural network 26*20x45*45
X = reshape(X(:), dim * dim, size(X,1)/dim * size(X,2)/dim)';

%check
tmp = reshape(X(1, :), dim, dim);
tmp = appendToRightColumn(tmp, 0);
tmp = appendToLowRow(tmp, 0);
tmp = tmp == check;
fprintf('Next output should be equal to: %d \n', (dim + 1) * (dim + 1));
sum(sum(tmp))
end
