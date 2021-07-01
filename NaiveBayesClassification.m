%-----------------------------------------------------------
% Naive Bayes Classification
%-----------------------------------------------------------
%
% Loading the dataset
dataset=readmatrix('data_banknote_authentication.txt');

%Getting column and row numbers
[row,column]=size(dataset);

%Takes the attributes except the Class column
x=dataset(:,1:column-1);

%Including only the class column
y=dataset(:,column);

%Counts every index of the dataset, 1st column is class "0", second column
%is class "1".
e(:,1)=find(y==0);
e(:,2)=find(y==1);

%separating each value of the class indices
%for class 0
w1=x(e(:,1),:);
%for class 1
w2=x(e(:,2),:);


%Calculating each class' prior probability by their lengths
%For our dataset they are all equal
p=[length(find(y==0)),length(find(y==1))];
p=p./row;

%means of each class
mean1=sum(w1)./length(w1);
mean2=sum(w2)./length(w2);

%temporary value for each class
%that calculates Centered Data Matrix for each of the classes
%After it uses those values to find standard deviation
var=w1;
for i=1:length(w1)
    var(i,:)=w1(i,:)-mean1;
end
%Standard Deviation of the 1st class (Sigma)
std1=sqrt((sum(var.^2))/length(w2));

%Centered Data Matrix for 2nd class
var=w2;
for i=1:length(w2)
    var(i,:)=w2(i,:)-mean2;
end
%Standard Deviation of the 2nd class (Sigma)
std2=sqrt((sum(var.^2))/length(w1));

%Asks for input from the user for the sample data in the form of [x,x,x,x]
x1 = input('Enter 4 numbers in brackets, in the form [x,x,x,x]:');

%Calculates the posterior probability of each class by their multiplication
% (The formula with the multiplication sign Pi)
%Function for the formula is at the end of this script.
p0=p(1)*formula(x1(1),mean1(1),std1(1))*formula(x1(2),mean1(2),std1(2))*formula(x1(3),mean1(3),std1(3))*formula(x1(4),mean1(4),std1(4));
p1=p(2)*formula(x1(1),mean2(1),std2(1))*formula(x1(2),mean2(2),std2(2))*formula(x1(3),mean2(3),std2(3))*formula(x1(4),mean2(4),std2(4));

%decides for the class
disp('For x=');
disp(x1);
if(p0>p1)
    disp('It is class 0');
elseif(p1>p0)
    disp('It is class 1');
end

%function for the formula
function [result] = formula(x,mean,std)
result=(1/(std*sqrt(2*3.14)))*exp(-(((x-mean)^2)/(2*(std^2))));
end
