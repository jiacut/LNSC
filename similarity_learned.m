function [Z, permed_index, epoch] = similarity_learned(data, num_samples, sigma, alpha)

disp('Randomly selecting samples...');
num_rows = size(data, 1);
permed_index = randperm(num_rows);
sample_data = data(permed_index(1:num_samples), :);
permed_data = data(permed_index, :);
clear data;

disp('Calculating static matrices...');
A = euclidean(sample_data', sample_data');
A = single(A); 
A = single(exp(-(A.*A) ./ (2*sigma*sigma)));%sigma太小的话会产生NAN或者INF，但是太大的话聚类结果会受到影响

B = euclidean(permed_data', sample_data');
B = single(B);
B = single(exp(-(B.*B) ./ (2*sigma*sigma)));

one_n = ones(num_rows, num_samples);
Y = one_n * 2 - 2 * B;

disp('Updating learned similarity matrix...');
epoch = 0;
max_epoch = 200;
gap = 1;
min_gap = 1e-4;

Z = B;
while (gap > min_gap) && (epoch < max_epoch)
    Zt = Z .* sqrt(B ./ (Z * A + alpha * Y)); % tr(Z^T * Y)
%     Zt = Z .* sqrt(B ./ (Z * A + alpha * Z)); % (|| Z ||_F)^2 = tr(Z^T * Z), easy convergence
%     Zt = Z .* sqrt(B ./ (Z * A)); % results like Gaussian kernel
	gap = sum(sum((Zt - Z).^2));
	Z = Zt;
    epoch = epoch + 1;
end
