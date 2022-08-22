function [Z, permed_index] = similarity(data, num_samples, sigma)
%   Input  : data           : n-by-d data matrix, where n is the number of data,
%                             d is the number of dimensions
%            num_samples : number of random samples m
%            sigma          : sigma value used in computing similarity
%
%   Output : Z : n-by-m similarity matrix
%            permed_index : new order index of data

disp('Randomly selecting samples...');
num_rows = size(data, 1);
permed_index = randperm(num_rows);
sample_data = data(permed_index(1:num_samples), :);
other_data = data(permed_index(num_samples+1:num_rows), :);
clear data;

disp('Calculating distance among samples...');
A = euclidean(sample_data', sample_data');
A = single(A);

disp('Calculating distance between samples and other points...');
B = euclidean(sample_data', other_data');
B = single(B);
clear sample_data other_data;

% S = exp^(-(dist^2 / 2*sigma^2))
disp('Converting distance matrix to similarity matrix...');
A = single(exp(-(A.*A) ./ (2*sigma*sigma)));
B = single(exp(-(B.*B) ./ (2*sigma*sigma)));
Z = [A B]';
