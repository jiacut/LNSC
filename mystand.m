function out=mystand(A)
out=[];
n=size(A,1);%��ȡ����
minA = min(A); %��ȡ��Сֵ
maxA = max(A);%��ȡ����ֵ
out = (A-repmat(minA,n,1))./repmat(maxA-minA,n,1);%ʹ��repmat��ÿ��Ԫ�ؽ����ظ������ǵ�����һ��Ҫ��./
end
