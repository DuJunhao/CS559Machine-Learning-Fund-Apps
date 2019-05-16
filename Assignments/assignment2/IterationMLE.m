function results=IterationMLE

accuracy=zeros(10,1);
for i=1:10
accuracy(i)=MLE();
end
meanOfAccuracy=mean(accuracy);
standardDerivation=std(accuracy);
results={meanOfAccuracy,standardDerivation};
