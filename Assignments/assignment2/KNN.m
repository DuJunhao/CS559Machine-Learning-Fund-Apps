function relustLabel = KNN(test,sample,labels,k)
%
%   test is from test dataset, sample is from the training dataset, labels
%   are classification from training data.
%

[row , col] = size(sample);
differenceMatrix = repmat(test,[row,1]) - sample ;
distanceMatrix = sqrt(sum(differenceMatrix.^2,2));
[B , IX] = sort(distanceMatrix,'ascend');
len = min(k,length(B));
relustLabel = mode(labels(IX(1:len)));
end
