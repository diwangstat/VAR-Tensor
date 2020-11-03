function RRR_select(ts,lag_order,max_rank)
    N,T = size(ts); P = lag_order; n = T-P;
    BIC_list = zeros(max_rank);

    Y = y[:,(P+1):T];
    X = y[:,P:(T-1)];
    for s = 2:P
        X = [X;y[:,(P-s+1):(T-s)]];
    end
    S_xx = X*X'; S_xy = X*Y'; S_yx = Y*X';
    M1 = S_yx*inv(S_xx)*S_xy;

    for r in 1:max_rank
        V = eigen(M1).vectors[:,(N-r+1):N];
        RRR = inv(S_xx)*S_xy*V*V';
        error_rrr = norm(Y-RRR'*X)/sqrt(n*N);
        df = (N+N*P-r)*r;
        BIC_list[r] = 2n*log(error_rrr^2)+(df+1)*log(n);
    end
    rank_select = findmin(BIC_list)[2];
    V = eigen(M1).vectors[:,(N-rank_select+1):N];
    RRR = inv(S_xx)*S_xy*V*V';

    return (A=RRR', rank=rank_select)
end
