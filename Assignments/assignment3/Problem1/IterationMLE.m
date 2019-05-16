
meanData=zeros(1,3);
accuracy=zeros(10,1);
for i=1:10
results=MLE();
accuracy(i)=results{1};
end
meanData(1,:)=results{2};
meanOfAccuracy=mean(accuracy);
result={meanOfAccuracy,meanData};
