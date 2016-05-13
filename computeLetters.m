function y = computeLetters(X, num_letters)
    y = zeros(size(X,1), 1);
    block_size = size(X,1) / num_letters;
    for letter = 1:num_letters,
        for blocki = 1:block_size,
            y(blocki + block_size * (letter - 1), 1) = letter;
        end
    end
end
