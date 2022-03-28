function [D, stopout] = MENLR(X, class_id, lambda)
%---------------Optimization problem--------------------------
%    min ||X*D-b1 -R||_F^2 + (lam_1/2)*(||A||_F^2 + ||B||_F^2) + (lam_2/2)*||D||_F^2       
%               s.t. D = AB, r_{i yi}-max_{j != yi} r_{ij}>=1.
%
%---------------Parameters-------------------------------------
% X -------- Centered training samples (remove the mean), each column is a sample
% trgnd ---- The label vector the training samples
% lambda --- Regularization paramter, uasually smaller than 1
%--------------- End ------------------------------------------

if nargin < 3 
    disp('Please input the data matrix M, the indicator matrix W and the rank r.');
end

[dim, N] = size(X);
num_class = numel(unique(class_id));
lambda1 = lambda;% Not sensitive to lambda1,  lambda1 = lambda2 for simplicity.
lambda2 = lambda;
r = min(5*max(floor((num_class-1)/5),1),num_class-1);

Y = zeros(num_class, N);
for i = 1 : N
    Y(class_id(i),  i) = 1.0;  
end

%% Initializing optimization variables
scale = 1e-5;   
[D0, b0] = LSR(X,  Y,  lambda2);
A = randn(dim,r)*scale;
B = randn(r,num_class)*scale;
C1 = randn(dim,num_class)*scale; % Lagrange multiplier
mu = 1e-8;
outIter = 35;

cnt = 0;
XXT = X*X';
%% Main algorithm
while cnt < outIter
        % update A
        if cnt == 0
            D = D0;
            b = b0;
        end
        Dt = D;
        mD = mu*D;
        A1 = C1+mD;
        A2 = lambda1*eye(r)+mu*(B*B');
        A  = (A1*B')/A2;

        % update B
        B = (lambda1*eye(r) + mu*(A'*A))\A'*mD;

        % optimize matrix R.
        P = X' * D  + ones(N, 1) * b';
        for ind = 1:N
            R(ind,:) = optimize_R(P(ind,:), class_id(ind));
        end

        % update D
        AB = A*B;
        TR = 2*X*R+mu*AB-C1;
        TL = 2*XXT + (lambda2+mu)*eye(dim);
        D = TL\ TR;
        bt = X'*D-Y';
        b = mean(bt',2);
            
        % update C1 (Lagrange multipliers)
        dD = D - AB;
        C1 = C1 + mu*dD;
        cnt = cnt + 1;
        stopout(cnt) = trace ( (Dt - D)' * (Dt - D) );
        disp(['iter = ' num2str(cnt), ' stop = ' num2str(stopout(cnt)) ]);
end
end

function [W, b] = LSR(X, Y, gamma)
    [dim, N] = size (X);
    [~, N] = size(Y);
    XMean = mean(X')';
    XX = X - repmat(XMean, 1, N);

    W = [];
    b = [];
    t0 =  XX * XX' + gamma * eye(dim);
    W = t0 \ (XX * Y');   
    b = Y - W' * X;
    b = mean(b')'; 
end

function T = optimize_R(R, label)
    classNum = length(R);
    T = zeros(classNum,1);
    V = R + 1 - repmat(R(label),1,classNum);    
    step = 0;
    num = 0;
    for i = 1:classNum
        if i~=label
            dg = V(i);
            for j = 1:classNum;
                if j~=label
                    if V(i) < V(j)
                        dg = dg + V(i) - V(j);
                    end
                end
            end
            if dg > 0
                step = step + V(i);
                num = num + 1;
            end
        end
    end
    step = step / (1+num);
    for i = 1:classNum
        if i == label
            T(i) = R(i) + step;
        else
            T(i) = R(i) + min(step - V(i), 0);
        end
    end
end
