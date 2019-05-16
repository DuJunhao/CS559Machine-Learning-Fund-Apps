function results=MLE
%get the data
data=csvread('pima-indians-diabetes.csv');
data_all=pca(data);
meanData=mean(data_all);
%get the training dataset and testing dataset
y=length(data_all);
c = cvpartition(y,'HoldOut',0.5);
% Now you can find the indices of your training and test sets
trainingIdx = training(c);
testIdx = test(c);
% Now you can find your training and test data
train_class_all = data_all(trainingIdx,:);
testData = data_all(testIdx,:);
%labels for training and test
train_label = data(trainingIdx,9);
text_label=data(testIdx,9);
[train_row,train_col]=size(train_class_all);
[m,n]=size(data_all);
%get the data with different labels
 for i=1:2
    train_classification{i}=train_class_all(find(train_label==i-1),:);%every data among feature2 to feature4
    NUM{i}=length(train_class_all(find(train_label==i-1),1));%number of every class
 end
%priori probabilities
for i=1:2
    PW{i}=NUM{i}/length(train_class_all);%the ratio
end
%the mean of MLE
for i=1:2
   train_mean{i}=(sum(train_classification{i}))/NUM{i};
   %train_mean{i}=mean(train_classification{i}); both are correct
end
% covariance matrix in the MLE
for i=1:2
        for x=1:n
            for y=1:n
             train_cov{i}(x,y)=(sum((train_classification{i}(:,x)-train_mean{i}(1,x)).*(train_classification{i}(:,y)-train_mean{i}(1,y))))/(NUM{i}-1);%covariance matrix
            end
        end
end

%train_cov{1}=cov(train_classification{1});
%train_cov{2}=cov(train_classification{2}); both are correct

 for i=1:2
     train_cov_inv{i}=inv(train_cov{i});%inverse of the covariance matrix
     train_cov_det{i}=det(train_cov{i});%determinant of the covariance matrix
 end
 %the codes below are the manual approach to get he inverse and determinant of this 3*3 matrix 
%  E=zeros(3,3);
%  for i=1:3
% E(i,i)=1;
%  end
% for i=1:2
%  train_cov_det{i}=train_cov{i}(1,1).*train_cov{i}(2,2).*train_cov{i}(3,3)+train_cov{i}(1,2).*train_cov{i}(2,3).*train_cov{i}(3,1)+train_cov{i}(1,3).*train_cov{i}(2,1).*train_cov{i}(3,2)-train_cov{i}(3,1).*train_cov{i}(2,2).*train_cov{i}(1,3)-train_cov{i}(3,2).*train_cov{i}(2,3).*train_cov{i}(1,1)-train_cov{i}(3,3).*train_cov{i}(2,1).*train_cov{i}(1,2);
%     train_cov_inv{i}=E/train_cov{i};
% end

%minimum erroe with bayes

%get the minimum error for every data in every classifier
 for i=1:2
     for j=1:length(testData)
     text_data_one=testData(j,:);
     g{j,i}=(-0.5)*(text_data_one-train_mean{i})*train_cov_inv{i}*(text_data_one'-train_mean{i}')-0.5*log(abs(train_cov_det{i}))+log(PW{i});
     end
 end
%remove the infinity
for j=1:length(testData)
    for i=1:2
        if abs(g{j,i})<=10000
         p{j,i}=g{j,i};
        end
    end
end
%get the maximum
nummax=zeros(length(testData),1);
for j=1:length(testData)
PRow=[p{j,:}];
nummax(j,1)=max(PRow);
end 
%mark the label
label=zeros(length(testData),1);
for j=1:length(testData)
switch nummax(j,1)
    case g{j,1} 
       label(j,1)=0;
    case g{j,2}
       label(j,1)=1;
end 
end
bo=zeros(length(testData),1);
for j=1:length(testData)  
if label(j,1)==text_label(j,1)
   bo(j)=1;
end
end

%get accuracy
correct=0;
for j=1:length(bo)
    if bo(j)==1
        correct=correct+1;
    end
end
accuracy=correct/length(testData);
results={accuracy,meanData};




