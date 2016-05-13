function m = appendToLowRow(matrix, number)
    m = [matrix ; ones(1, size(matrix, 2)) * number];
end
