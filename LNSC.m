clear

 load data/USPS.mat
%load data/tr45.mat

%[fea,gnd]=dataset_num(10);

data =fea;
label =gnd;
%data=mystand(X);
%data(data==Inf)=0;
%data(isnan(data))=0;
num_clusters = length(unique(label));
num_samples = 1000;
sigma =50;
alpha =0.7;
tic;
%[Z, permed_index] = similarity(data, num_samples, sigma);
[Z, permed_index, epoch] = similarity_learned(data, num_samples, sigma, alpha);
cluster_labels = nystrom(Z, permed_index, num_clusters);

result = ClusteringMeasure(label, cluster_labels);
t=toc;
 
dlmwrite('USPS.txt',[alpha sigma num_samples result t],'-append','delimiter','\t','newline','pc');
% USPS£º
% similarity: num_samples = 1000, sigma = 50
% result = 0.7102    0.6614    0.7819
% similarity_learned: USPS, num_samples = 1000, sigma = 50, alpha = 0.8, 
%result= 0.7965    0.8160    0.8589  1.363 
% MNIST£º
% similarity: num_samples = 1000, sigma = 3000
% result = 0.5503    0. 4787    0.5996
% similarity_learned: num_samples = 1000, sigma = 3000, alpha = 0.8
% result = 0.6198    0.6280    0.6863   44.96