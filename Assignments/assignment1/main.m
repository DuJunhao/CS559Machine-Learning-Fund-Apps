close all; 
clear all;
clc
data1=part1(1,2,2000);
data2=part1(4,3,1000);
data3=[data1;data2];
figure(1)
histfit(data1,2000);
figure(2)
histfit(data2,1000);
figure(3)
histfit(data3,3000);

[f, xi]=ksdensity(data1);
figure(4)
fit(xi',f','gauss1');
plot(xi,f);

