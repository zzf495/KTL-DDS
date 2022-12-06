%% Demo of DDSA
%% Information
%%%         Knowledge Transfer Learning via Dual Density Sampling for Resource-Limited Domain Adaptation
%%%         Author          ZeFeng Zheng et al.
%% Input
%%%      proxyFunction          The function to be executed
%%%      alpha                  The smooth coefficient
%%%      k                      The number of neighbors
%%%      k2                     The selection rate (20% default)
%%%      classifier             Indicate which classifier is used
%%%                              0   -   NCM
%%%                              1   -   SVM
%% Output
%%%      acc                    The classification accuracy (number,0~1)
%%%      acc_ite                The classification accuracy in each iteration (list)
clc; clear all;
addpath(genpath('./util/'));
path='./data/Office31/office-';
suffix='-resnet50-noft.mat';
domains1 = {'A','A','D','D','W','W'};
domains2 = {'D','W','A','W','A','D'};
result=[];
accIteration=[];
%% Parameter Setting
options= defaultOptions(struct(),...
                        'alpha',0.9,...   
                        'k',10,...          
                        'k2',0.2,...        
                        'classify',1,...
                        'display',1);
options.proxyFunction=@func.LDA_DA;     % A proxy function
optsPCA.ReducedDim=512;                 % The dimension reduced before training
for i = 1:6
    %% Load data
    src = [path domains1{i} suffix];
    tgt = [path domains2{i} suffix];
    fprintf('%d: %s_vs_%s\n',i,domains1{i},domains2{i});
    load(src);
    %%% Load Xs
    feas = resnet50_features;
    feas = feas ./ repmat(sum(feas,2),1,size(feas,2));
    Xs=double(zscore(feas,1))';
    Ys = double(labels'+1);
    %%% Load Xt
    load(tgt);
    feas = resnet50_features;
    feas = feas ./ repmat(sum(feas,2),1,size(feas,2));
    Xt=double(zscore(feas,1))';
    Yt = double(labels'+1);
    %% Run PCA to reduce the dimensionality
    domainS_features_ori=Xs';domainT_features=Xt';
    X = double([domainS_features_ori;domainT_features]);
    P_pca = PCA(X,optsPCA);
    domainS_features = domainS_features_ori*P_pca;
    domainT_features = domainT_features*P_pca;
    %% Run KTL-DDS
    [acc,acc_ite]=DDSA(Xs,Ys,Xt,Yt,options);
    accIteration=[accIteration;acc_ite];
    result(i)=acc;
end
fprintf('Mean accuracy: %.4f\n',mean(result));
