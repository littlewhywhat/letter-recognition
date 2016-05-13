%load test_data
X_test = loadLetters('test_set.bmp', 46); 
%load train_data
X_train = loadLetters('train_set.bmp', 46);
%remove redundant columns
%clean = removeRedund([X_test; X_train]);
%X_test = clean(1 : size(X_test, 1), :);
%X_train = clean(size(X_test, 1) + 1 : size(clean, 1), :);
clear clean;
