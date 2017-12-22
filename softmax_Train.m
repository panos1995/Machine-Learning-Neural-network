function [W1,W2] = softmax_Train(T, X, lambda, Winit_1,Winit_2, options,iterations)

W1=Winit_1;
W2=Winit_2;

K = size(W1,1);
L= size(W2,1);
% Maximum number of iteration of gradient ascend
iter = iterations; 

% Tolerance
tol = options(1);

% Learning rate
eta = options(2);
 
Ewold = -Inf; 

for it=1:iter

    [Ew,grad_W1,grad_W2] = costgrad(W1,W2, X, T, lambda);
    fprintf('Iteration: %d, Cost function: %f\n',it, Ew);
    
      if abs(Ew - Ewold) < tol 
        break;
    end
    
    % Update parameters based on gradient ascent 
    W2 = W2 + eta*grad_W2;
    W1 = W1 + eta*grad_W1; 
    
    Ewold = Ew; 



end