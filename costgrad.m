function [Ew,grad_W1,grad_W2] = costgrad(W1,W2, X, T, lambda)

K = size(W1,1);
L= size(W2,1);

%paragontes der gia W(2)
a1=X*W1';%N*M
Z=cos(a1);%N*M
z=size(Z,1);
Z=[ones(z,1),Z];%N*M+1
a2=Z*W2';%N*K

%paragontes gia der tou W(1)
W2_without_bias = W2;
W2_without_bias(:,1)=[]; %SIZE W2 WITHOUT K *M.




M = max(a2, [], 2); %it takes the max across rows of the matrix a2

Ew = sum(sum( T.*a2 )) - sum(M)  - sum(log(sum(exp(a2 - repmat(M, 1, L)), 2)))  - (0.5*lambda)*(sum(sum(W1.*W1))+sum(sum(W2.*W2)));

if nargout>1
    %softmax
   Y=softmax(a2);
   %grad for W1
   grad_W2=((T-Y)')*Z - lambda*W2;
   %grad for W2
   inner_sum=(T-Y)*W2_without_bias; %(N x K )* (K x M)--> N* M 
   inner_sum_multi_h= inner_sum.*(-sin(a1)); % NxM *(ELEMENT) N x M-->NxM
   grad_W1=inner_sum_multi_h'*X -lambda*W1; %(M * N) * (N * D+1) --> M x D+1
end