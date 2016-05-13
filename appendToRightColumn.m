function m = appendToRightColumn(matrix, number)
    m = [ matrix, ones(size(matrix,1), 1) * number ];
end
