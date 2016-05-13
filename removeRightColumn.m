function m = removeRightColumn(matrix) 
    m = matrix(:, 1 : size(matrix, 2) - 1);
end
