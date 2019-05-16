%function result=KNNdatgingTest
data=csvread('pima-indians-diabetes.csv');
dataMat = data(:,2:4);

len = size(dataMat,1);
k = 11;
% ratio of datas
Ratio = 0.5;
numTest = Ratio * len;
%normalization
maxV = max(dataMat);
minV = min(dataMat);
range = maxV-minV;
newdataMat = (dataMat-repmat(minV,[len,1]))./(repmat(range,[len,1]));
accuracy=zeros(10,1);


% test
for index=1:10
    
% First you make crossvalidation partitioning on your data
% y is a vector which contains the categories of your observations
% 'HoldOut' an optional property to make training and test set
% Fraction of data to form test set
y=length(newdataMat);
c = cvpartition(y,'HoldOut',0.5);
% Now you can find the indices of your training and test sets
trainingIdx = training(c);
testIdx = test(c);
% Now you can find your training and test data
trainingData = newdataMat(trainingIdx,:);
testData = newdataMat(testIdx,:);

error=0;
labels = data(trainingIdx,9);
    for i = 1:numTest
        classifyresult = KNN(testData(i,:),trainingData,labels,k);
        fprintf('results:%d  real results are:%d\n',[classifyresult labels(i)])
        if(classifyresult~=labels(i))
            error = error+1;
        end
    end
  accuracy(index,1)=1-error/numTest;
  fprintf('accuracy:%f\n',1-error/numTest);
end

meanOfAccuracy=mean(accuracy);
standardDerivation=std(accuracy);
result={meanOfAccuracy,standardDerivation};
    
