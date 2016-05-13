function m = resh(matrix, dim) 
    m = matrix(1:dim, :);
    shift = dim;
    for i = 2:size(matrix,1)/dim
        m = [m , matrix(shift + 1:shift+dim, :)]; 
        shift = shift + dim;
    end
end
