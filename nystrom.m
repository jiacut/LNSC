function [cluster_labels, evd_time, total_time] = nystrom(Z, permed_index, num_clusters)
%   Input  : Z              : n-by-m learned similarity matrix, where n is the number of data,
%                             m is the number of random samples
%            permed_index  : new order index of data
%            num_clusters   : number of clusters
%
%   Output : cluster_labels : n-by-1 vector containing cluster labels
%            evd_time       : running time for eigendecomposition
%            kmeans_time : running time for k-means
%            total_time     : total running time

disp('Normalizing Z for Laplacian...');
tic;
[n, m] = size(Z);
A = Z(1:m, :); % 矩阵A由m个样本点之间的成对相似性构成
A = single(A);
B = Z((m+1):n, :); % 矩阵B由n-m个剩余点和m个样本点之间的成对相似性构成
B = single(B);

B_T = B';
d1 = sum(A, 2) + sum(B_T, 2);
d2 = sum(B, 2) + B * (pinv(A) * sum(B_T, 2));
dhat = sqrt(1./[d1; d2]); % D^(-1/2)
% Z = diag(dhat) * Z; % Z = D^(-1/2) * Z
Z = repmat(dhat, 1, m) .* Z;
Z = single(Z);
clear B B_T d1 d2 dhat;
time1 = toc;

disp('Orthogalizing and eigendecomposition...');
Asi = sqrtm(pinv(A)); % A^(-1/2)
% Calculate M = A^(-1/2) * Z' * Z * A^(-1/2)
M = Asi * Z' * Z * Asi;
M = (M + M')/2; % Make sure M is symmetric, sometimes M can be non-symmetric because of numerical inaccuracy

[U L] = eig(M);
[val ind] = sort(diag(L), 'descend');
U = U(:, ind); % in decreasing order
L = L(ind, ind); % in decreasing order
clear A M;

% Calculate orthogonal eigenvector V = Z * A^(-1/2) * U * L^(-1/2)
V = Z * Asi * U(:, 1:num_clusters) * pinv(sqrt(L(1:num_clusters, 1:num_clusters)));
clear Z Asi L U;
time2 = toc;

disp('Performing kmeans...');
% Normalize each row to be of unit length
sq_sum = sqrt(sum(V.*V, 2)) + 1e-20;
Vn = V ./ repmat(sq_sum, 1, num_clusters);
clear sq_sum V;
cluster_labels = k_means(Vn, [], num_clusters);
% Restore cluster_labels in original order
cluster_labels(permed_index) = cluster_labels;
clear permed_index;
total_time = toc;

% Calculate and show time statistics
evd_time = time2 - time1
kmeans_time = total_time - time2
total_time
disp('Finished!');
