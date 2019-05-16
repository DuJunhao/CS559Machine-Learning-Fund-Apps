function [w y1 y2 Jw] = FisherLinearDiscriminat(data, label)
% FLD Fisher Linear Discriminant.
% data : D*N data
% label : {+1,-1}
% Reference:M.Bishop Pattern Recognition and Machine Learning p186-p189

% compute means and scatter matrix
%-------------------------------

inx1 = find( label == 1);
inx2 = find( label == -1);
n1 = length(inx1);
n2 = length(inx2);

m1 = mean(data(:,inx1),2);
m2 = mean(data(:,inx2),2);

S1 = (data(:,inx1)-m1*ones(1,n1))*(data(:,inx1)-m1*ones(1,n1))';
S2 = (data(:,inx2)-m2*ones(1,n2))*(data(:,inx2)-m2*ones(1,n2))';
Sw = S1 + S2;

% compute FLD 
%-------------------------------
W = inv(Sw)*(m1-m2);

y1 = W'*m1;  %label=+1
y2 = W'*m2;  %label=-1
w = W;
Jw = (y1-y2)^2/(W'*Sw*W);
