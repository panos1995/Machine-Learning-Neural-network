clear all; 
close all; 

load mnist_all.mat;
K = 10;
T = []; 
X = [];
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
figure; 
hold on; 
for j=1:10

    s = ['train' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);   
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1; 
    X = [X; Xtmp]; 
    T = [T; Ttmp]; 
    
    s = ['test' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp]; 
   
    % plot some training data
    ind = randperm(size(Xtmp,1));
    for i=1:10
        subplot(10,10,10*(j-1)+i);     
        imagesc(reshape(Xtmp(ind(i),:),28,28)');
        axis off;
        colormap('gray');     
    end
 
end
X = X/255; 
Xtest = Xtest/255; 

[N D] = size(X);

% Add 1 as the first for both the training input and test inputs 
X = [ones(sum(Ntrain),1), X];
Xtest = [ones(sum(Ntest),1), Xtest]; 
M=[100,200,300,400,500];
% Regularization parameter lambda 
lambda = [0.5,0.4,0.3,0.6,0.7,1]; 
% Maximum number of iterations of the gradient ascend
iterations=[500,600,700,800];
% Tolerance 
options(1) = 1e-6; 
% Learning rate 
options(2) = 0.5/N;  
%runs for every M
error=zeros(1,size(M,2)*size(lambda,2)*size(iterations,2));
for i=1:size(M,2)
    for j=1:size(lambda,2)
        for k=1:size(iterations,2)
disp(['M = : ']);
disp(i);
W_one_init = 0.001*randn(M(i),D+1);
W_two_init = 0.001*randn(K,M(i)+1);




 

% Do a gradient check first
% (in a small random subset of the data so that 
% the gradient check will be fast)
W1 = randn(size(W_one_init)); 
W2 = randn(size(W_two_init)); 
ch = randperm(N); 
ch = ch(1:20);
[grad_W1,grad_W2, numgradEw_1,numgradEw_2] = gradcheck_neural(W1,W2,X(ch,:),T(ch,:),lambda(j)); 




% Train the model 
[W_one,W_two] = softmax_Train(T, X, lambda(j), W_one_init,W_two_init, options,iterations(k)); 

[Ttest, Ytest]  = ml_softmaxTest(W_one,W_two, Xtest); 

[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;
disp(['The error of the method is: ' num2str(err)])
error(1,i*j*k)=err;

header1 = 'M';
header2 = 'lamda';
head2 = 'iter';
headerr= 'error';
%fid=fopen('MyErrorFileFinal.txt','a');
%fprintf(fid, [ header1 ' ' header2 ' ' head2 ' ' headerr '\n']);
%fprintf(fid, '%f %f %f %f \n', [M(i) lambda(j) iterations(k) err]');
%fclose(fid);
        end
    end
end

