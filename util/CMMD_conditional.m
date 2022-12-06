function Mc = CMMD_conditional(X1,X4,X2,X3,Y1,Y4,Y2,Y3,gamma,alpha,C)
%% Inputs:
%%%     Xsg                 The source global sample set, m * n_{s,g}
%%%     Xsl                 The source local sample set, m * n_{s,l}
%%%     Xtg                 The target global sample set, m * n_{t,g}
%%%     Xtl                 The target local sample set, m * n_{t,l}
%%%     gamma, beta         The hyper-parameters
%%%     C                   The number of labels
%% Output:
%%%     Mc                  The conditional distribution matrix
%% numbers
n1=size(X1,2);
n2=size(X4,2);
n3=size(X2,2);
n4=size(X3,2);
n=n1+n2+n3+n4;
Mc=zeros(n,n);
for c = 1:C
    c1=find(Y1==c);[n1,c1]=removeErr(n1,c1);
    c2=find(Y4==c);[n2,c2]=removeErr(n2,c2);
    c3=find(Y2==c);[n3,c3]=removeErr(n3,c3);
    c4=find(Y3==c);[n4,c4]=removeErr(n4,c4);
    cY=[c1;n1+c2;n1+n2+c3;n1+n2+n3+c4];
    nc=length(cY);
    nc_Xs=length(c1)+length(c2);
    nc_Xt=length(c3)+length(c4);
    if nc_Xt==0||nc_Xs==0
       continue; 
    end
    e=zeros(nc,1);
    e(1:nc_Xs)=gamma/length(c1);
    if n2>0
        pos=length(c1)+1;
        e(pos:nc_Xs)=(1-gamma)/length(c2);
    end
    if n3>0
        pos=nc_Xs+1;
        e(pos:pos+length(c3))=-alpha/length(c3);
    end
    if n4>0
        pos=nc_Xs+1+length(c3);
        e(pos:end)=-(1-alpha)/length(c4);
    end
    if length(find(cY))==length(find(e))
        Mc(cY,cY)=e*e';
    end
end
end

function [n,c]=removeErr(n,c)
    if isempty(c)
       n=0;
       c=[];
    end
end