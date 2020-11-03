# orthogonal and sparse regression by ADMM
function lasso_ortho_admm(Y,X,lambda,B_col,B_row,ϵ=1e-8,max_itr=100)
    r = min(B_col,B_row); n = length(Y);

    # initialize
    beta =  inv(X'*X)*(X'*Y);
    W = reshape(beta,B_row,B_col);
    M = zeros(B_row,B_col);

    rho = 1; gamma = 1;
    Gram = inv(X'*X/n+rho*eye(B_col*B_row)+gamma*eye(B_col*B_row));
    XYprod = X'*Y/n;

    for i = 1:max_itr
        # step 1. Update beta without orthogonal constraint
        gamma = 1; Q = reshape(beta,B_row,B_col); Z = zeros(B_row,B_col);

        WmM = rho*reshape(W-M,B_col*B_row,1);
        for oth_loop in 1:max_itr
            beta = Gram*(XYprod+WmM+gamma*reshape(Q-Z,B_col*B_row,1));
            s_temp = svd(reshape(beta,B_row,B_col)+Z);
            Q = s_temp.U[:,1:r]*s_temp.V[:,1:r]';
            Z = Z+reshape(beta,B_row,B_col)-Q;

            if sum((reshape(beta,B_row,B_col)-Q).^2)<ϵ*sum(Q.^2) break end
        end

        # step 2. Update w by soft-thresholding
        W = (reshape(beta,B_row,B_col).+M.-(2*lambda)/rho.>0).*(reshape(beta,B_row,B_col).+M.-(2*lambda)/rho)-(-reshape(beta,B_row,B_col).-M.-(2*lambda)/rho.>0).*(-reshape(beta,B_row,B_col).-M.-(2*lambda)/rho);

        # step 3. Dual update
        M = M+reshape(beta,B_row,B_col)-W;

        # stopping criteria
        if sum((reshape(beta,B_row,B_col)-W).^2) < sum(reshape(beta,B_row,B_col).^2)*ϵ break end

    end
    return W
end
