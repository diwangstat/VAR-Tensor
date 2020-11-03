# Matrix nuclear norm

function MNN_VAR(ts,lag_order,n_lambda,ϵ=1e-5,max_itr=100)

    K,T = size(ts); P = lag_order;
    Y = ts[:,(P+1):T]'; global X = ts[:,P:(T-1)]';
    for s = 2:P
        global X = [X';ts[:,(P-s+1):(T-s)]]';
    end
    n = T-P;

    # OLS initalization
    A_init = (X\Y)*1;

    # lambda path
    lambda_max = log(n_lambda);
    lambda_seq = exp.(log(lambda_max).-log(n_lambda)/n_lambda*(0:n_lambda-1));

    A = A_old = W = A_init;
    C = reshape(zeros(K^2*P),K*P,K);
    rho = 1;

    output_A = Array{Array{Float64}}(undef,n_lambda);
    output_sigma_hat = zeros(n_lambda);
    output_AIC = zeros(n_lambda);
    output_BIC = zeros(n_lambda);

    for i_lambda = 1:n_lambda;
        lambda = lambda_seq[n_lambda-i_lambda+1];
        for i = 1:max_itr
            A = inv(X'X/n + rho*eye(K*P))*(X'*Y/n+rho*(W-C));
            S1 = svd(A+C);
            d1 = (S1.S.-lambda/(2*rho)); d1 = (abs.(d1)+d1)/2;
            W = S1.U*Diagonal(d1)*S1.Vt;
            C = C + A - W;

            # breaking rule
            if max(norm(A-W),norm(A-A_old))<ϵ*norm(A) break end
            A_old = A;
            output_A[i_lambda] = A;
            sigma_hat = sqrt(norm(Y-X*A)^2/(n*K));
            output_sigma_hat[i_lambda] = sigma_hat;
            r = sum(svd(A).S.>=1e-1); df = (2*K-r)*r;
            AIC = (n*K)*log(sigma_hat^2)+df*2;
            BIC = (n*K)*log(sigma_hat^2)+df*log(n);
            output_AIC[i_lambda] = AIC;
            output_BIC[i_lambda] = BIC;
        end
    end
    return (A = output_A, sigma_hat = output_sigma_hat, AIC = output_AIC, BIC = output_BIC)
end

function TNN_VAR(ts,lag_order,n_lambda,lambda_max = log(n_lambda),ϵ=1e-5,max_itr=100)

    K,T = size(ts); P=lag_order;
    Y = ts[:,(P+1):T]'; global X = ts[:,P:(T-1)]';
    for s = 2:P
        global X = [X';ts[:,(P-s+1):(T-s)]]';
    end
    n = T-P;

    # OLS initalization
    A_init = (X\Y)'*1;
    A_init_Tnsr = matten(A_init,1,[K,K,P]);

    # lambda path
    lambda_max = log(n_lambda);
    lambda_seq = exp.(log(lambda_max).-log(n_lambda)/n_lambda*(0:n_lambda-1));

    A = A_old = A_init_Tnsr;
    W = Array{Array{Float64}}(undef,3);
    C = Array{Array{Float64}}(undef,3);
    for k = 1:3
        W[k] = A_init_Tnsr;
        C[k] = matten(reshape(zeros(K*K*P),K,K*P),1,[K,K,P]);
    end

    output_A = Array{Array{Float64}}(undef,n_lambda);
    output_sigma_hat = zeros(n_lambda);
    output_AIC = zeros(n_lambda);
    output_BIC = zeros(n_lambda);

    rho = 1;

    for i_lambda = 1:n_lambda
        lambda = lambda_seq[n_lambda-i_lambda+1];
        for i = 1:max_itr
            # A step
            global W_C = matten(reshape(zeros(K*K*P),K,K*P),1,[K,K,P]);
            for k = 1:3
                global W_C = W_C + W[k] - C[k];
            end
            A = matten((inv(X'X/n + 3*rho*eye(K*P))*(X'*Y/n+rho*tenmat(W_C,1)'))',1,[K,K,P]);

            # W & C step
            for k = 1:3
                Svd = svd(tenmat(A+C[k],k));
                d1 = (Svd.S .- (lambda/(3))/(2*rho));
                d1 = (abs.(d1)+d1)/2;
                W[k] = matten(Svd.U*Diagonal(d1)*Svd.Vt,k,[K,K,P]);
                C[k] = C[k] + A - W[k];
            end

            # breaking rule
            dif = zeros(3);
            for k = 1:3
                dif[k] = norm(A-W[k]);
            end
            if maximum([maximum(dif),norm(A-A_old)])<ϵ*norm(A) break end
            A_old = A;
        end
        output_A[i_lambda] = A;
        sigma_hat = sqrt(norm(Y-X*tenmat(A,1)')^2/(n*K));
        output_sigma_hat[i_lambda] = sigma_hat;
        s = zeros(3); df = 0;
        for k = 1:3
            s[k] = sum(svd(tenmat(A,k)).S.>=1e-1);
        end
        df = (K+K*P-s[1])*s[1]+(K+K*P-s[2])*s[2]+(P+K^2-s[3])*s[3];

        AIC = (n*K)*log(sigma_hat^2)+(df+1)*2;
        BIC = (n*K)*log(sigma_hat^2)+(df+1)*log(n);
        output_AIC[i_lambda] = AIC;
        output_BIC[i_lambda] = BIC;
    end
    return (A = output_A, sigma_hat = output_sigma_hat, AIC = output_AIC, BIC = output_BIC)
end
