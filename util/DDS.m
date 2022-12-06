function [localDelta,localIndicater,localhotmatrixY,...
    globalDelta,globalIndicater,globalhotmatrixY]       = DDS(X,Y,C,k,k2)
%%%  Select representative samples with the top-k highest densities
%%  Input
%%%
%%%     X                  The sample set with m * n
%%%     Y                  The (pseudo)labels of X with n * 1
%%%     k                  The number of neighbors
%%%     k2                 The selection rate
%%  Output
%%%     localDelta         The local density of X
%%%
%%%     localIndicater     The indicator matrix that indicates which sample
%%%                        is selected
%%%
%%%     localhotmatrixY    An one-hot matrix of selected sample w.r.t local
%%%                        density
%%%     globalDelta        The global density of X
%%%
%%%     globalIndicater    The indicator matrix that indicates which sample
%%%                        is selected
%%%
%%%     globalhotmatrixY   An one-hot matrix of selected sample w.r.t global
%%%                        density

        nt=size(X,2);
        if 0<k2&&k2<=1
           k2=floor(k2*nt/C);
           k2=max(k2,1);
%            fprintf('k2 number: %2d/%2d \n',k2,nt);
        end
        opt.Metric='Euclidean';
        opt.WeightMode='HeatKernel';
        opt.NeighborMode='KNN';
        opt.bNormalized=1;
        opt.k=0;
        [localDelta,~,~,~]=getDensity(X,Y,k,opt);
        [globalDelta,~,~,~]=getDensity(X,Y,nt,opt);
        localIndicater = getDensityIndicater(X,Y,C,k2,localDelta);
        globalIndicater = getDensityIndicater(X,Y,C,k2,globalDelta);
        localhotmatrixY=localIndicater;
        localhotmatrixY=localhotmatrixY./repmat(sum(localhotmatrixY,1),nt,1);
        globalhotmatrixY=globalIndicater;
        globalhotmatrixY=globalhotmatrixY./repmat(sum(globalhotmatrixY,1),nt,1);
end

