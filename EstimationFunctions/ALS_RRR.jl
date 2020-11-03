function ALS_RRR(ts,lag_order,r,ϵ=1e-8,max_itr=30)

    K,N = size(ts); P = lag_order;
    Y = ts[:,P+1:N];
    X = ts[:,P:N-1];
    for s = 2:P
        X = [X;ts[:,(P-s+1):(N-s)]];
    end
    n = N-P;

    # Initialize: OLS for VAR
    A_OLS = (X*Y')'*inv(X*X');
    error_old = norm(Y-A_OLS*X)/sqrt(K*(N-P));

    # HOSVD decomposition
    Svd = svd(A_OLS);
    U = Svd.U[:,1:r]; V = diagm(Svd.S[1:r])*Svd.Vt[1:r,:];

    # Alternating least squares updating
    Y_col = reshape(Y,K*n,1);
    for itr = 1:max_itr
        # update U
        new_X = V*X;
        U = (new_X*Y')'*inv(new_X*new_X');

        # update V
        X_2 = Array{Float64}(undef,n*K,K*P*r);
        for i = 1:n
            X_2[((i-1)*K+1):(i*K),:] =  kron(X[:,i]',U)
        end
        V = reshape(inv(X_2'*X_2)*(X_2'*Y_col),r,K*P);

        # stopping rule
        error_new = norm(Y-U*V*X)/sqrt(K*(N-P));
        if (abs(error_new-error_old)<ϵ)
            break
        else
            error_old = error_new;
        end
    end

    return U*V
end
