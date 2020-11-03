# Sparse higher-order reduced-rank estimator

function SHORR_sep(ts,lag_order,mtl_rank,n_lambda,init_value="MLR",ϵ=1e-6,max_itr=50)

    K,N = size(ts); P = lag_order;
    Y = ts[:,(P+1):N];
    global X = ts[:,P:(N-1)];
    rho = 1;
    for s = 2:P
        global X = [X;ts[:,(P-s+1):(N-s)]];
    end
    n = N - P;

    rrr_est = ALS_MLR(ts,lag_order,mtl_rank);
    G = rrr_est.G;
    U = rrr_est.U;

    # lambda_path
    lambda_max = 0.1;
    lambda_seq = exp.(log(lambda_max).-log(n_lambda)/n_lambda*(0:n_lambda-1))/2;

    A_old = rrr_est.A;

    output_coef_tnsr = Array{Array{Float64}}(undef,n_lambda);
    output_G = Array{Array{Float64}}(undef,n_lambda);
    output_U = Array{Array{Array{Float64}}}(undef,n_lambda);
    output_sigma_hat = zeros(n_lambda);
    output_AIC = zeros(n_lambda);
    output_BIC = zeros(n_lambda);
    output_loop = zeros(n_lambda);

    for u_lambda = 1:n_lambda

        rho = 1; loop = 1
        for i_admm = 1:max_itr
            # update U1
            X_1 = kron((tenmat(G,1)*kron(U[3],U[2])'*X)',eye(K));
            U[1] = lasso_ortho_admm(reshape(Y,K*n,1),X_1,lambda_seq[n_lambda-u_lambda+1],mtl_rank[1],K);

            # update U2
            X_2 = Array{Float64}(undef,n*K,K*mtl_rank[2]);
            G1 = tenmat(G,1);
            for i = 1:n
                X_2[((i-1)*K+1):(i*K),:] = U[1]*G1*kron((reshape(X[:,i],K,P)*U[3])',eye(mtl_rank[2]));
            end
            U[2] = lasso_ortho_admm(reshape(Y,n*K,1),X_2,lambda_seq[n_lambda-u_lambda+1],K,mtl_rank[2])';

            # update U3
            X_3 = Array{Float64}(undef,n*K,P*mtl_rank[3]);
            for i = 1:n
                X_3[((i-1)*K+1):(i*K),:] = U[1]*G1*kron(eye(mtl_rank[3]),U[2]'*reshape(X[:,i],K,P));
            end
            U[3] = lasso_ortho_admm(reshape(Y,n*K,1),X_3,lambda_seq[n_lambda-u_lambda+1],mtl_rank[3],P);

            # update G
            X_g = Array{Float64}(undef,n*K,prod(mtl_rank));
            for i = 1:n
                X_g[((i-1)*K+1):(i*K),:] =  kron(X[:,i]'*kron(U[3],U[2]),U[1]);
            end
            G = reshape(inv(X_g'*X_g)*(X_g'*reshape(Y,n*K,1)),mtl_rank[1],mtl_rank[2],mtl_rank[3]);

            # stopping criteria
            A_new = ttm(G,U,[1,2,3]);
            if norm(A_new-A_old)<norm(A_old)*ϵ break end
            A_old = A_new;
            rho = rho * 2;
            loop = loop + 1;
        end
        output_loop[u_lambda] = loop;
        A_new = ttm(G,U,[1,2,3]);
        df = sum(abs.(U[1]).>1e-5)+sum(abs.(U[2]).>1e-5)+sum(abs.(U[3]).>1e-5)+sum(abs.(G).>1e-5);
        sigma_hat = sqrt(norm(Y-tenmat(A_new,1)*X)^2/(n*K));
        AIC = (n*K)*log(sigma_hat^2)+df*2;
        BIC = (n*K)*log(sigma_hat^2)+df*log(n*K);
        output_coef_tnsr[u_lambda] = A_new;
        output_G[u_lambda] = G;
        output_U[u_lambda] = U;
        output_AIC[u_lambda] = AIC;
        output_BIC[u_lambda] = BIC;
        output_sigma_hat[u_lambda] = sigma_hat;
    end
    return (A = output_coef_tnsr, G = output_G, U = output_U, sigma_hat = output_sigma_hat, AIC = output_AIC, BIC = output_BIC, itr = output_loop)
end
