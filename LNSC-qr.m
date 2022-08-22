clear

load data/USPS.mat
% data = X;
% label = y;
%load data/Postures.mat


%[fea,gnd]=dataset_num(10);

data =fea;
label =gnd;
%data(data==Inf)=0;
%data(isnan(data))=0;
num_clusters = length(unique(label));
num_samples = 1000;
sigma = 50;   
alpha = 0.8;
tic
%[Z, permed_index] = similarity(data, num_samples, sigma);
[Z, permed_index, epoch] = similarity_learned(data, num_samples, sigma, alpha);
% [Z, permed_index] = similarity2(data, num_samples, sigma, 'uni-sample');
% [Z, permed_index, epoch] = similarity_learned2(data, num_samples, sigma, alpha, 'uni-sample');

%cluster_labels = nystrom(Z, permed_index, num_clusters);
cluster_labels_qr = nystrom_qr(Z, permed_index, num_clusters);
t=toc;
%result = ClusteringMeasure(label, cluster_labels);
result_qr = ClusteringMeasure(label, cluster_labels_qr);
fprintf('result:\t%12.6f %12.6f %12.6f \n',[result_qr]);
dlmwrite('USPS-qr.txt',[alpha sigma num_samples result_qr t],'-append','delimiter','\t','newline','pc');
% USPS
% similarity: num_samples = 1000, sigma = 50
% result = 0.7089    0.6602    0.7806
% result_qr = 0.7112    0.6638    0.7824

% similarity_learned: num_samples = 1000, sigma = 50, alpha = 0.8
% result = 0.7819    0.8212    0.8627
% result_qr = 0.916    0.8396    0.916

% MNIST
% similarity: num_samples = 1000, sigma = 3000
% result = 0.5464    0.4764    0.6038
% result_qr = 0.6128    0.5061    0.6427

% similarity_learned: num_samples = 1000, sigma = 3000, alpha = 0.8
% result = 0.6518    0.6369    0.7042
% result_qr = 0.6622    0.6471    0.7144
