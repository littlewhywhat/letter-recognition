function processData(inputfile, outputfile, num_labels, dim) 

X = loadLetters(inputfile, dim); 

y = computeLetters(X, num_labels);

X = double(X)/255;

save(outputfile, 'X', 'y');
end
