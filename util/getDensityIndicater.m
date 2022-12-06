function [indicater] = getDensityIndicater(Xs,Ys,C,k,delta)

%     densityX=[];
%     densityXc={};
    [m,ns]=size(Xs);
    indicater=zeros(ns,C);
    for i=1:C
       idx=find(Ys==i);
       deltaC=delta(idx);
       minK=min(k,length(idx));
       [~,deltaIdx]=sort(deltaC,'descend');
%        densityXc{i}=Xs(:,idx(deltaIdx(1:minK)));
%        densityX=[densityX,densityXc{i}];
       indicater(idx(deltaIdx(1:minK)),i)=1;
    end
end

