function out=mystand(A)
out=[];
n=size(A,1);%获取行数
minA = min(A); %获取极小值
maxA = max(A);%获取极大值
out = (A-repmat(minA,n,1))./repmat(maxA-minA,n,1);%使用repmat对每个元素进行重复处理，记得这里一定要用./
end
