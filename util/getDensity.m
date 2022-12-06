function [delta,graph,sortGraph,sortIdx]=getDensity(X,Y,k,opt)
%%%     Calculate the global/local density of the sample
%%  Input
%%%
%%%     X                  The sample set with m * n
%%%     Y                  The (pseudo)labels of X with n * 1
%%%     k                  The number of neighbors
%%%
%%  Output
%%%
%%%     delta              The density of X

        n=size(X,2);
        C=length(unique(Y));
        delta=zeros(n,1);
        % similarity
        knnResults=zeros(n,k);
        dist_Knn=zeros(n,k);
        % dissimilarity in all
        dissimilarKnnResults=zeros(n,k);
        dist_dissimilarKnn=zeros(n,k);
        % get lapgraph
        graph=lapgraph(X',opt);
        [sortGraph,sortIdx]=sort(graph,2,'descend');
        similarPensity=zeros(n,1);
        disSimilarPensity=zeros(n,1);
        for i=1:n
        	nY=Y(sortIdx(i,:));
            %% similarity
            realIdx=find(nY==Y(i));
            minK=min(k+1,length(realIdx));
            knnResults(i,1:minK)=realIdx(1:minK);
            dist_Knn(i,1:minK)=sortGraph(i,realIdx(1:minK));
            similarPensity(i)=1/(minK-1)*sum(dist_Knn(i,2:end),2);
            %% dissimilarity
            realIdx=find(nY~=Y(i));
            minK=min(k+1,length(realIdx));
            dissimilarKnnResults(i,1:minK)=realIdx(1:minK);
            dist_dissimilarKnn(i,1:minK)=sortGraph(i,realIdx(1:minK));
            disSimilarPensity(i)=1/(minK-1)*sum(dist_dissimilarKnn(i,2:end),2);
        end
        delta=similarPensity-disSimilarPensity;
end