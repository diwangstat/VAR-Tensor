# Data-driven SHORR estimation
# Two-step parameter tuning
# 1. rank selection via regularization + truncation
# 2. lambda tuning via information criterion

function SHORR_select(ts,lag_order,n_lambda,ϵ=1e-6,max_itr=50)

    # rank selection
    (K,T) = size(ts); P = lag_order;

    MNN_est = MNN_VAR(ts,lag_order,30);
    MNN_select = matten((MNN_est.A[findmin(MNN_est.AIC)[2]])',[1],[2,3],[K,K,P]);
    ranks = RidgeRankSelect(MNN_select,sqrt(K*P*log(T)/(10*T)));

    # check rank
    while maximum(ranks)^2 > prod(ranks)
        ranks[findmin(ranks)[2]] = ranks[findmin(ranks)[2]]+1
    end

    # SHORR estimation
    SHORR_est = SHORR(ts,P,ranks,n_lambda);
    return (A = SHORR_est.A, rank = ranks, G = SHORR_est.G, U = SHORR_est.U, AIC = SHORR_est.AIC, BIC = SHORR_est.BIC)
end
