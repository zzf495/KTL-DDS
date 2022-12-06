function M0 = CMMD_marginal(Xsg,Xsl,Xtg,Xtl,gamma,alpha,C)
%% Inputs:
%%%     Xsg                 The source global sample set, m * n_{s,g}
%%%     Xsl                 The source local sample set, m * n_{s,l}
%%%     Xtg                 The target global sample set, m * n_{t,g}
%%%     Xtl                 The target local sample set, m * n_{t,l}
%%%     gamma, beta         The hyper-parameters
%%%     C                   The number of labels
%% Output:
%%%     M0                  The marginal distribution matrix
%% numbers
n1=size(Xsg,2);
n2=size(Xsl,2);
n3=size(Xtg,2);
n4=size(Xtl,2);
n=n3+n4;
e = [gamma/(n1)*ones(n1,1);...
    (1-gamma)/(n2)*ones(n2,1);...
    -alpha/n3*ones(n3,1);...
    -(1-alpha)/n4*ones(n4,1);...
    ];
M0 = e * e' * C;  %multiply C for better normalization
end

