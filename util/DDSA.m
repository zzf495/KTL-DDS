function [acc,acc_ite] = DDSA(Xs,Ys,Xt,Yt,options)
%% Input
%%%      proxyFunction          The function to be executed
%%%      alpha                  The smooth coefficient
%%%      k                      The number of neighbors
%%%      k2                     The selection rate (20% default)
%%%      classifier             0: NCM, 1: SVM
%% Output
%%%      acc                    The classification accuracy (number,0~1)
%%%      acc_ite                The classification accuracy in each iteration (list)
    options=defaultOptions(options,...
                            'alpha',0.9,...    % weight of local / global
                            'k',7,...          % KD  => k
                            'k2',32,...        % The selection rate
                            'classify',1);
    proxy=options.proxyFunction;
    alpha=options.alpha;
    k=options.k;
    k2=options.k2;
    C=length(unique(Ys));
    Xs=normr(Xs')';
    Xt=normr(Xt')';
    if options.classify==1
       knn_model = fitcknn(Xs',Ys,'NumNeighbors',1);
       Ytpseudo = knn_model.predict(Xt');
    else
        % get Ytpseudo
        svmmodel = train(double(Ys), sparse(double(Xs')),'-s 1 -B 1.0 -q');
        %%% the Yt below is used to calculate the accuracy only,
        %%% Yt is not involved in the training.
        [Ytpseudo,~,~] = predict(double(Yt), sparse(double(Xt')), svmmodel,'-q');
    end
    fprintf('[init] acc: %.4f\n',getAcc(Ytpseudo,Yt));
    
    %% Xs density
    [~,Es,~, ~,GEs,~] = DDS(Xs,Ys,C,k,k2);
    [~,Et,~, ~,GEt,~] = DDS(Xt,Ytpseudo,C,k,k2);
    Es=logical(sum(Es,2));
    GEs=logical(sum(GEs,2));
    Et=logical(sum(Et,2));
    GEt=logical(sum(GEt,2));
    Xs_traing=[Xs(:,Es),Xs(:,GEs)];
    Ys_train=[Ys(Es);Ys(GEs)];
    Xt_traing=[Xt(:,Et),Xt(:,GEt)];
    Yt_train=[Yt(Et);Yt(GEt)];
    %% execute Proxy
    [acc_ite,YtpseudoFromProxy] = proxy(Xs_traing,Ys_train,Xt_traing,Yt_train,options);
    %%% KD & LP
    manifold.k = 10;
    manifold.Metric = 'Cosine';
    manifold.WeightMode = 'Cosine';
    manifold.NeighborMode = 'KNN';
    W=lapgraph([Xt_traing,Xt]',manifold);
    if size(YtpseudoFromProxy,2)>1
       YtpseudoFromProxy=YtpseudoFromProxy'; 
    end
    Y=hotmatrix([YtpseudoFromProxy;Ytpseudo],C,0);
    [~,Ytpseudo] = classifyLP(Y,W,alpha);
    Ytpseudo=Ytpseudo(length(YtpseudoFromProxy)+1:end);
    acc=getAcc(Ytpseudo,Yt);
    fprintf('Adapt acc:%.4f\n',acc);
    acc_ite=[acc_ite,acc];
end

