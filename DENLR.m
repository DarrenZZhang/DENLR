function [D, stopout] = DENLR(X, trgnd, lambda)
%---------------Optimization problem--------------------------
%    min ||X*D-(Y+E.*M)||_F^2 + (lambda1/2)*(||A||_F^2 + ||B||_F^2) + (lambda2/2)*||D||_F^2       
%                              s.t. D = AB, M>=0
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
num_class = numel(unique(trgnd));
lambda1 = lambda;% Not sensitive to lambda1,  lambda1 = lambda2 for simplicity.
lambda2 = lambda;
r = min(5*max(floor((num_class-1)/5),1),num_class-1);

Y = zeros(num_class, N);
E = -1 * ones(N, num_class);

for i = 1 : N
    Y(trgnd(i),  i) = 1.0;  
    E( i, trgnd(i) )  = 1.0;
end

%% Initializing optimization variables
scale = 1e-5;   
[D0, b] = LSR(X,  Y,  lambda2);
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
        end
        Dt = D;
        mD = mu*D;
        A1 = C1+mD;
        A2 = lambda1*eye(r)+mu*(B*B');
        A  = (A1*B')/A2;

        %----------- update B----------------
        B = (lambda1*eye(r) + mu*(A'*A))\A'*mD;

        %----------- update M----------------
        P = X' * D  - Y'; 
        M = optimize_M(P, E);

        %----------- update D----------------
        AB = A*B;
        TR = 2*X*(Y'+(E .* M))+mu*AB-C1;
        TL = 2*XXT + (lambda2+mu)*eye(dim);
        D = TL\ TR;

        %----------- update C1----------------
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

function M = optimize_M(P, B)
N = size(P, 1);
num_class = size(B, 2);

M1 = zeros(N, num_class);
M = max( B .* P,  M1); 
end
