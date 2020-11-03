# Alternating least squares algorithm for multilinear low-rank estimator
function ALS_MLR(ts,lag_order,mtl_rank,ϵ=1e-8,max_itr=30)

    K,N = size(ts); P = lag_order;
    Y = ts[:,P+1:N];
    X = ts[:,P:N-1];
    for s = 2:P
        X = [X;ts[:,(P-s+1):(N-s)]];
    end
    n = N-P;

    # Initialize: OLS for VAR
    A_OLS = (X*Y')'*inv(X*X');
    A_OLS_Tnsr = matten(A_OLS,[1],[2,3],[K,K,P]);
    error_OLS = norm(Y-A_OLS*X)/sqrt(K*(N-P))

    # HOSVD decomposition
    A_OLS_HOSVD = hosvd(A_OLS_Tnsr,reqrank=mtl_rank);
    G = A_OLS_HOSVD.cten;
    U = A_OLS_HOSVD.fmat;
    error_old = norm(Y-U[1]*tenmat(G,1)*kron(U[3],U[2])'*X)/sqrt(K*n)

    # Alternating least squares updating
    Y_col = reshape(Y,K*n,1);
    for itr = 1:max_itr
        # update U1
        new_X = tenmat(G,row=[1],col=[2,3])*kron(U[3]',U[2]')*X;
        U[1] = (new_X*Y')'*inv(new_X*new_X');

        # update U2
        X_2 = Array{Float64}(undef,n*K,K*mtl_rank[2]);
        G1 = tenmat(G,row=[1],col=[2,3]);
        for i = 1:n
            X_2[((i-1)*K+1):(i*K),:] =  U[1]*G1*kron((reshape(X[:,i],K,P)*U[3])',eye(mtl_rank[2]));
        end
        U[2] = reshape((Y_col'*X_2)*inv(X_2'*X_2),mtl_rank[2],K)';

        # update U3
        X_3 = Array{Float64}(undef,n*K,P*mtl_rank[3]);
        for i = 1:n
            X_3[((i-1)*K+1):(i*K),:] =  U[1]*G1*kron(eye(mtl_rank[3]),(U[2]'*reshape(X[:,i],K,P)));
        end
        U[3] = reshape(inv(X_3'*X_3)*(X_3'*Y_col),P,mtl_rank[3]);

        # update G
        X_g = Array{Float64}(undef,n*K,prod(mtl_rank));
        for i = 1:n
            X_g[((i-1)*K+1):(i*K),:] =  kron(X[:,i]'*kron(U[3],U[2]),U[1]);
        end
        G = reshape(inv(X_g'*X_g)*(X_g'*Y_col),mtl_rank[1],mtl_rank[2],mtl_rank[3]);

        # stopping rule
        error_new = norm(Y-U[1]*tenmat(G,1)*kron(U[3],U[2])'*X)/sqrt(K*(N-P));
        if (abs(error_new-error_old)<ϵ)
            break
        else
            error_old = error_new;
        end
    end

    # HOSVD to make U's orthonormal
    A_mlr = ttm(G,U,[1,2,3]);
    A_mlr_hosvd = hosvd(A_mlr,reqrank=mtl_rank);

    return (A=A_mlr, G=A_mlr_hosvd.cten, U=A_mlr_hosvd.fmat, error_mlr=error_old, error_ols=error_OLS)
end

function ALS_MLR_select(ts,lag_order,max_rank)
    N,T = size(ts); P = lag_order; n = T-P;
    BIC_list = reshape(zeros(prod(max_rank)),max_rank);
    for i in 1:max_rank[1]
        for j in 1:max_rank[2]
            for k in 1:max_rank[3]
                if maximum([i,j,k])^2 <= prod([i,j,k])
                    MLR = ALS_MLR(ts,P,[i,j,k]);
                    df = i*j*k+(N-i)*i+(N-j)*j+(P-k)*k;
                    BIC_list[i,j,k] = 2n*log(MLR.error_mlr^2)+(df+1)*log(n);
                else
                    BIC_list[i,j,k] = 10^15;
                end
            end
        end
    end
    rank_select = [findmin(BIC_list)[2][1],findmin(BIC_list)[2][2],findmin(BIC_list)[2][3]];
    MLR_select = ALS_MLR(ts,lag_order,rank_select);
    return (A=MLR_select.A, G=MLR_select.G, U=MLR_select.U, rank=rank_select)
end
