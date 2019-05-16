
meanData=zeros(1,8);
accuracy=zeros(10,1);
for i=1:10
results=fisherAndMLE();
accuracy(i)=results{1};
end
meanData(1,:)=results{2};
OptimalDirection=results{3};
meanOfAccuracy=mean(accuracy);
result={meanOfAccuracy,meanData};
