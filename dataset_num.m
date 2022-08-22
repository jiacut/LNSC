function[X,Y]=dataset_num(a)
if a==10
    load data/USPS.mat
    X = fea;
    Y =gnd;
    [~, col]=size(X);
    X1=sum(X, 2);
    X2=repmat(X1, 1, col);
    X=single(X);
    X=X./X2;
%     X(X==Inf)=0;
%     X(isnan(X))=0;
end
end

