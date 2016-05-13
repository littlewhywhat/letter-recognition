function m = resh2(matrix, dim) 
    m = matrix(:, 1:dim);
    shift = dim;
    for i = 2 : size(matrix, 2)/dim
        m = [m ; matrix(:, shift + 1 : shift + dim)];
        shift = dim + shift;
    end
end
