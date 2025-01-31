function build_A(A)
    Ni, Nj = size(A)[end-1:end]
    return [A[:,:,:,:,:,i,j] for i = 1:Ni, j = 1:Nj]
end

function build_M(A)
    D = size(A[1], 1)
    ap = [reshape(ein"abcde,fghmn->afbgchdmen"(A, conj(A)), D^2,D^2,D^2,D^2, 2,2) for A in A]
    M  = [ein"abcdee->abcd"(ap) for ap in ap]
    return ap, M
end