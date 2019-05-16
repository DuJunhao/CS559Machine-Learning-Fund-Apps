function [w, num_misclassified,wrongPoints,rightPoints] = perceptron(X,t)
% The perceptron algorithm 
%   X : D*N data
%   t : {+1,-1} labels
%   
%   w : [w0 w1 w2]   
%   mis_class : the wrongly classified points


%  check the label
data=X';
if size(unique(t))~=2
    return
elseif max(t)~=1
    return
elseif min(t)~=-1
    return
end

[dim num_data] = size(X);
w = ones(dim+1,1);%%w = [w0 w1 w2]'
X = [ones(1,num_data); X];
maxiter = 100000;
num_misclassified = 0;
iter = 0;

while iter<maxiter
    iter = iter+1;
    y = w'*X;
    label = ones(1, num_data);%{+1,-1}
    label(y<=0) = -1;  
    WrongIndex = find(label~=t); % the wrongly classified points
     RightIndex= find(label==t);
    num_misclassified = numel(WrongIndex); % the number of wrongly classified points
    if num_misclassified==0
        break
    end
    for i = 1:num_misclassified
        w = w + X(:,WrongIndex(i))*t(WrongIndex(i));
    end
end
if iter==maxiter
    disp(['reach the maximum iteration:' num2str(maxiter)])
end
wrongPoints=zeros(length(WrongIndex),2);
rightPoints=zeros(length(data)-length(WrongIndex),2);
for i=1:length(WrongIndex)
    wrongPoints(i,:)=data(WrongIndex(i),:);
end
for i=1:length(RightIndex)
    rightPoints(i,:)=data(RightIndex(i),:);
end
