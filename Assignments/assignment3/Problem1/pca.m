function pcaData=pca(X)

X=X(:,1:8);
Z=zscore(X);%normalize
[Row Col]=size(X);
R=corrcoef(Z);%get the correlation of the data and we can use the cov function to original data if we don't do the normalization
[V,D]=eig(R); %get the eigenValue D and eigenVector V
Cols=size(X,2);%get the number of features
characters=zeros(1,Cols);%keep eigenValue in the characters matrix
for i=1:Cols
    characters(i)=D(i,i);
end
[sortedCharacters,label]=sort(characters,'descend'); %sort and keep the result and index.
fprintf('                       Eigenvalue of the Correlation Matrix                          \n');
fprintf('          Eigenvalue     Difference       Proportion     Cumulative                  \n');
total=0;
for i=1:Cols
    fprintf(['    Z' num2str(i)]);
    fprintf('     %.4f',sortedCharacters(i));
    if i==Cols
        fprintf('            .  ');
    else
        fprintf('         %.4f',sortedCharacters(i)-sortedCharacters(i+1));
    end
    fprintf('           %.4f',sortedCharacters(i)/sum(sortedCharacters));
    total=total+sortedCharacters(i);
    fprintf('         %.4f\n',total/sum(sortedCharacters));
end
fprintf('                            Eigenvectors                          \n');
fprintf('             Z1             Z2             Z3             Z4              \n');
for i=1:Cols
    fprintf(['    X' num2str(i)]); %[]concat the string
    for j=1:Cols
        fprintf('   %8.4f    ',V(i,label(j))); %print
    end
    fprintf('\n');
end

k=3;

meanX=mean(Z);                                  %get the mean from the data
%all the data-mean and then multiply the eigenVector to get the principal component Score
tempX= repmat(meanX,Row,1);
SCORE=(Z-tempX)*V;                             %pca:SCORE
pcaData=SCORE(:,1:3);

