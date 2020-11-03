# Ridge type rank selection

function RidgeRankSelect(est,c)

    d = ndims(est);
    p = size(est);

    if d == 3
        S1 = svd(tenmat(est,1)).S;
        S2 = svd(tenmat(est,2)).S;
        S3 = svd(tenmat(est,3)).S;

        r1 = findmax((S1[1:(p[1]-1)].+c)./(S1[2:p[1]].+c))[2];
        r2 = findmax((S2[1:(p[2]-1)].+c)./(S2[2:p[2]].+c))[2];
        r3 = findmax((S3[1:(p[3]-1)].+c)./(S3[2:p[3]].+c))[2];

    end
    return(rank = [r1,r2,r3])

end
