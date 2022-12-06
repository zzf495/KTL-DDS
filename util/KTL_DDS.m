function [acc,acc_ite,Zs,Zt,Ytpseudo,P] = KTL_DDS(Xs,Ys,Xt,Yt,options)
%% Input
%%%
%%%      T                      The number of iterations
%%%      dim                    The reduced dimension
%%%      mu                     The tradeoff parameter between marginal and conditional distributions 
%%%      gamma                  The tradeoff parameter between Xsg and Xtg
%%%      alpha                  The tradeoff parameter between Xtg and Xtl
%%%      lambda                 The weight of regularization
%%%      eta                    The weight of manifold regularization
%%%      k                      The number of neighbors
%%%      k2                     The selection rate (20% default)
%%%      classifier             0: NCM, 1: SVM
%%%
%% Output
%%%      acc                    The classification accuracy (number,0~1)
%%%      acc_ite                The classification accuracy in each iteration (list)
%%%      Zs                     The source sample set after projection
%%%      Zt                     The target sample set after projection
%%%      Ytpseudo               The predicted labels of target sample set
%%%      P                      The projection matrix
    options=defaultOptions(options,...
                            'T',10,...          % The number of iterations
                            'dim',30,...        % The reduced dimension
                            'mu',0.5,...        % The tradeoff parameter between marginal and conditional distributions 
                            'gamma',0.1,...     % The tradeoff parameter between Xsg and Xtg
                            'alpha',0.9,...     % The tradeoff parameter between Xtg and Xtl
                            'lambda',0.1,...    % The weight of regularization
                            'eta',5,...         % The weight of manifold regularization
                            'k',7,...           % The number of neighbors
                            'k2',0.2,...        % The selection rate (20% default)
                            'classifier',0);    % 0: NCM, 1: SVM
    alpha=options.alpha;
    gamma=options.gamma;
    eta=options.eta;
    lambda=options.lambda;
    mu=options.mu;
    T=options.T;
    dim=options.dim;
    k=options.k;
    k2=options.k2;
    acc_ite=[];
    Xs=normr(Xs')';
    Xt=normr(Xt')';
    [X,ns,nt,n,m,C] = datasetMsg(Xs,Ys,Xt,1);
    %%% Init Ytpseudo
    if options.classifier==1
        knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
        Ytpseudo = knn_model.predict(Xt');
    else
        svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
        [Ytpseudo,~,~] = predict(zeros(nt,1), sparse(double(Xt')), svmmodel,'-q');
    end
    if isfield(options,'display')
        fprintf('[init] acc: %.4f\n',getAcc(Ytpseudo,Yt)); 
    end
    %%% Use DDS (Definition 6) to select representative samples from Xs and Xt
    [~,Es,~, ~,GEs,~] = DDS(Xs,Ys,C,k,k2);
    [~,Et,~, ~,GEt,~] = DDS(Xt,Ytpseudo,C,k,k2);
    %% Solve Definition 7 by Eqs. (12)-(14)
    Y=[Ys;Ytpseudo];
    flag11=sum(Es,2)>0;flag21=sum(Et,2)>0;
    flag12=sum(GEs,2)>0;flag22=sum(GEt,2)>0;
    %%% Local Xs
    Xsl=Xs(:,flag11);
    Ysl=Ys(flag11);
    %%% Local Xt
    Xtl=Xt(:,flag21);
    %%% Global Xs
    Xsg=Xs(:,flag12);
    Ysg=Ys(flag12);
    %%% Global Xt
    Xtg=Xt(:,flag22);
    %%% \hat_X, i.e., global & local + Xs & Xt
    Xp=[Xsg,Xsl,Xtg,Xtl];
    %%% The mariginal condition of CMMD computed by Eq. (13)
    M0=CMMD_marginal(Xsg,Xsl,Xtg,Xtl,gamma,alpha,C);
    %%% Centering matrix
    H=centeringMatrix(n);
    XHX=X*H*X';
    %%% Solve KD by Eqs. (15) and (16)
    manifold.k = 10;
    manifold.Metric = 'Cosine';
    manifold.WeightMode = 'Cosine';
    manifold.NeighborMode = 'KNN';
    Lt=computeL(Xt,manifold);
    Lt=Lt./norm(Lt,'fro');
    L=Xt*Lt*Xt';
    for i=1:T
        Y2=Ytpseudo(flag22);
        Y3=Ytpseudo(flag21);
        %%% The conditional condition of CMMD computed by Eq. (13)
        Mc = CMMD_conditional(Xsg,Xsl,Xtg,Xtl,Ysg,Ysl,Y2,Y3,gamma,alpha,C);
        M=mu*M0+(1-mu)*Mc;
        M=M./norm(M,'fro');
        M=Xp*M*Xp';
        %% learn projection
        [P,~]=eigs(M+eta*L+lambda*eye(m),XHX,dim,'sm' );
        P=real(P);
        Zk=P'*X;
        Zk=L2Norm(Zk')';
        Zs=double(Zk(:,1:ns));
        Zt=double(Zk(:,ns+1:end));
        %%% predict by NCM or SVM
        if options.classifier==1
            knn_model = fitcknn((Zs*hotmatrix(Ys,C,1))',1:C,'NumNeighbors',1);
            Ytpseudo = knn_model.predict((Zt'));
        else
            svmmodel = train(double(Ys), sparse(double(Zs')),'-s 1 -B 1.0 -q');
            [Ytpseudo,~,~] = predict(zeros(nt,1), sparse(double(Zt')), svmmodel,'-q');
        end
        acc=getAcc(Ytpseudo,Yt);
        acc_ite=[acc_ite,acc];
        if isfield(options,'display')
           fprintf('[%2d] acc: %.4f\n',i,acc); 
        end
    end
end

