function [grad_W1,grad_W2, numgradEw_1,numgradEw_2] = gradcheck_neural(W1,W2,X,T,lambda) 
%
%

[M,D] = size(W1);
[K,EM] = size(W2);
% Compute the analytic gradient 
[Ew,grad_W1,grad_W2] = costgrad(W1,W2, X, T, lambda);

% Ccan all parameters to compute 
% numerical gradient estimates
epsilon = 1e-6; 
%gia W1
numgradEw_1 = zeros(M,D); 
numgradEw_2= zeros(K,EM);
for m=1:M
    for d=1:D
        Wtmp = W1; 
        Wtmp(m,d) = Wtmp(m,d) + epsilon; 
        Ewplus = costgrad(Wtmp,W2, X, T, lambda); 
        
        Wtmp = W1; 
        Wtmp(m,d) = Wtmp(m,d) - epsilon; 
        Ewminus = costgrad(Wtmp,W2, X, T, lambda);
        
        numgradEw_1(m,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end

%gia W2
for m=1:K
    for d=1:EM
        Wtmp = W2; 
        Wtmp(m,d) = Wtmp(m,d) + epsilon; 
        Ewplus = costgrad(W1,Wtmp, X, T, lambda); 
        
        Wtmp = W2; 
        Wtmp(m,d) = Wtmp(m,d) - epsilon; 
        Ewminus = costgrad(W1,Wtmp, X, T, lambda);
        
        numgradEw_2(m,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end
% Display the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff = abs(grad_W1 - numgradEw_1); 
diff_2=abs(grad_W2 - numgradEw_2); 
disp(['The maximum abolute norm in the gradcheck_W1 is ' num2str(max(diff(:))) ]);
disp(['The maximum abolute norm in the gradcheck_W2 is ' num2str(max(diff_2(:))) ]);
end