# require SparseArrays

function permutation_matrix(r_vec,k)
    r1 = r_vec[1]; r2 = r_vec[2]; r3 = r_vec[3];
    v1 = Array(1:(r1*r2*r3));
    vk = reshape(tenmat(matten(reshape(v1,r1,r2*r3),1,r_vec),k),r1*r2*r3);
    return sparse(vk,v1,ones(r1*r2*r3))
end
