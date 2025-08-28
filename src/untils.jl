Zygote.@adjoint function LinearAlgebra.svd(A)
    res = LinearAlgebra.svd(A)
    res, function (dy)
        dU, dS, dVt = dy
        return (svd_back(res.U, res.S, res.V, dU, dS, dVt === nothing ? nothing : dVt'),)
    end
end

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F.^ 2 .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end


"""
    takagi_decomposition(M; D_trunc)

Perform Takagi factorization for complex symmetric matrices. 
Decomposes matrix `M` into `M = A * transpose(A)` where:
- `M` is a complex symmetric matrix (M = M^T)
- `D_trunc`: truncation dimension to retain in the decomposition

This implementation:
1. Diagonalizes M†M to find eigenvectors (V) and eigenvalues (λ)
2. Extracts singular values from diagonal matrix D = VᵀMV
3. Constructs matrix A using first D_trunc eigenvectors scaled by sqrt(singular values)

Primarily used for SU parameterization in iPEPS tensor network simulations.
"""
function takagi_decomposition(M; D_trunc)
    norm(M - transpose(M)) < 1e-10 || throw(ArgumentError("M should be a complex symmetric matrix"))
    # Diagonalize M†M and sort eigenvectors by descending eigenvalues
    _, V = eigen(M'*M; sortby=x->-x)
    
    # Calculate diagonal matrix of singular values squared (D = VᵀMV)
    D = diag(transpose(V) * M * V)
    
    # Construct decomposition matrix: A = V*_trunc * sqrt(D_trunc)
    A = conj(V[:,1:D_trunc]) * diagm(sqrt.(D[1:D_trunc]))
    
    return A
end
