function [Ttest, Ytest]  = ml_softmaxTest(W1,W2, Xtest) 
%function [Ttest, Ytest]  = ml_softmaxTest(W, Xtest) 
%  
% What it does: It tests an already trained logistic regression model
%
% Inputs: 
%         W: the K x (D+1) dimensional matrix of the parameters   
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         Test:  Ntest x 1 vector of the predicted class labels
%         Ytest: Ntest x K matrix of the sigmoid probabilities     
%
% Michalis Titsias (2014)

% Mean predictions
a1=Xtest*W1';%N*M
Z=cos(a1);%N*M
z=size(Z,1);
Z=[ones(z,1),Z];%N*M+1
a2=Z*W2';%N*K
Ytest = softmax(a2);

% Hard classification decisions 
[~,Ttest] = max(Ytest,[],2);