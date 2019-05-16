
X={[-2,1],[-5,4],[-3,1],[0,-3],[-8,-1],[2,5],[1,0],[5,-1],[-1,-3],[6,1]};
data=zeros(length(X),2);
Firstlabel=zeros(length(X),1);
for i=1:length(X)
    data(i,:)=X{i};
    if i<6
        Firstlabel(i,:)=-1;
    end
    if i>5
         Firstlabel(i,:)=1;
    end
end

[w y1 y2 Jw]=FisherLinearDiscriminat(data',Firstlabel);
[w2, num_misclassified,wrongPoints,rightPoints]  = perceptron(data',Firstlabel');