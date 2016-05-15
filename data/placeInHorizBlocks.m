function m = placeInHorizBlocks(matrix, block_size) 
    m = matrix(1:block_size, :);
    shift = block_size;
    for i = 2 : size(matrix, 1)/block_size
        m = [m , matrix(shift + 1:shift + block_size, :)]; 
        shift = shift + block_size;
    end
end
