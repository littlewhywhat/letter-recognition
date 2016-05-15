function m = removeRedund(matrix)
    rem = [];
    for j = 1:size(matrix, 2),
        col = matrix(1, j); 

        cnt = 0;
        for i = 2: size(matrix, 1),
            if matrix(i,j) != col,
                cnt = 1;
                continue;
            end
        end
        if (cnt == 0)
            rem = [rem, j];
        end
    end
    m = matrix;
    m(:, rem) = []; 
end
