function C4v_restriction(A)
    A += permutedims(conj(A), (1,4,3,2,5,6)) # up-down
    A += permutedims(conj(A), (3,2,1,4,5,6)) # left-right
    A += permutedims(conj(A), (2,1,4,3,5,6)) # diagonal
    A += permutedims(conj(A), (4,3,2,1,5,6)) # rotation

    return A
end
"""
    _restriction_ipeps(ipeps)
```
        4
        │
 1 ── ipeps ── 3
        │
        2
```
"""
function _restriction_ipeps(A)
    return A
end

function central_canonical1(A)
    Ac, Rs = pepsgeneral(A)
    A = ARstoA1(Ac, Rs)
    return A
end

function central_canonical2(A)
    Ac, Rs = pepsgeneral(A)
    A = ARstoA2(Ac, Rs)
    return A
end

function pepsgeneral(A; tol=1e-12)
    function leftorth(A)
        D,d,N = size(A)[[1,5,6]]
        Q, R = TeneT.qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4, 6)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D, N), (1, 2, 3, 5, 4, 6))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5, 6))
    end

    D = size(A, 1)
    Rs = Zygote.Buffer([_arraytype(A)(rand(eltype(A),D,D)) for _ in 1:4])
    for i in 1:4
        A, R = leftorth(A)
        A = rotate(A)
        Rs[i] = R
    end
    conv = Inf
    i = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            Rs[i] = R * Rs[i] 
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum 
        i += 1
        if i > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A, copy(Rs)
end

function ARstoA1(A, Rs)
    # R31 = ein"ab,cb->ac"(Rs[3], Rs[1])
    @tensor R31[a,c] := Rs[3][a,b] * Rs[1][c,b]
    F = svd(R31)
    sqrtS = diagm(sqrt.(F.S))
    # @show F.S
    Rs3 = F.U * sqrtS
    Rs1 = conj(F.V * sqrtS)

    # R42 = ein"ab,cb->ac"(Rs[4], Rs[2])
    @tensor R42[a,c] := Rs[4][a,b] * Rs[2][c,b]
    F = svd(R42)
    # @show F.S
    sqrtS = diagm(sqrt.(F.S))
    Rs4 = F.U * sqrtS
    Rs2 = conj(F.V * sqrtS)

    # return ein"(((abcdpn,dh),ae),bf),cg->efghpn"(A, Rs1,Rs2,Rs3,Rs4)
    return @tensor Aout[e,f,g,h,p,n] := A[a,b,c,d,p,n] * Rs1[d,h] * Rs2[a,e] * Rs3[b,f] * Rs4[c,g]
end

function ARstoA2(A, Rs)
    # R3 = ein"ab,cb->ac"(Rs[3], Rs[1])
    # R4 = ein"ab,cb->ac"(Rs[4], Rs[2])
    @tensor R3[a,c] := Rs[3][a,b] * Rs[1][c,b]
    @tensor R4[a,c] := Rs[4][a,b] * Rs[2][c,b]

    # return ein"(abcdpn,bf),cg->afgdpn"(A, R3,R4)
    return @tensor Aout[a,f,g,d,p,n] := A[a,b,c,d,p,n] * R3[b,f] * R4[c,g]
end

function guage_transfer(A, G)
    G1, G2 = G
    # expG1 = exp(G1)
    # expG2 = exp(G2)
    # expG1inv = exp(-G1)
    # expG2inv = exp(-G2)
    expG1 = G1
    expG2 = G2
    expG1inv = inv(G1)
    expG2inv = inv(G2)
    return @tensor out[e,f,g,h,p,n] := A[a,b,c,d,p,n] * expG1[e,a] * expG1inv[c,g] * expG2[h,d] * expG2inv[b,f]
end

function find_local_min_norm_G(A)
    atype = _arraytype(A)
    A_cpu = Zygote.@ignore Array(A) 

    f(G) = norm(guage_transfer(A_cpu, G))
    function fg(G)
        cost, vjp = pullback(f, G)
        g = vjp(1)[1]
        return cost, g
    end 
    @info "optimize gauge to minimize norm"
    @info "initial norm = $(norm(A))"
    D = size(A, 1)
    eltypeA = eltype(A)

    G = atype.(optimize(fg, [randn(eltypeA,D,D), randn(eltypeA,D,D)], LBFGS(maxiter=100))[1])
    return G
end

function local_min_norm(A)
    G = Zygote.@ignore find_local_min_norm_G(A)
    AG = guage_transfer(A, G)
    @info "final norm = $(norm(AG))"
    
    return AG
end

function pepsgeneral_Ac(A; tol=1e-12)
    function leftorth(A)
        D,d,N = size(A)[[1,5,6]]
        Q, R = TeneT.qrpos(reshape(permutedims(A, (1, 2, 3, 5, 4, 6)), D^3 * d, D))
        Q = permutedims(reshape(Q, D, D, D, d, D, N), (1, 2, 3, 5, 4, 6))
        return Q, R
    end

    function rotate(A)
        return permutedims(A, (2, 3, 4, 1, 5, 6))
    end

    D = size(A, 1)
    # Rs = Zygote.Buffer([_arraytype(A)(rand(eltype(A),D,D)) for _ in 1:4])
    for i in 1:4
        A, R = leftorth(A)
        A = rotate(A)
        # Rs[i] = R
    end
    conv = Inf
    i = 0
    while conv > tol
        conv_sum = Zygote.@ignore 0
        for i in 1:4
            A, R = leftorth(A)
            A = rotate(A)
            # Rs[i] = R * Rs[i] 
            conv_sum += norm(R - one(R))
        end
        conv = conv_sum 
        i += 1
        if i > 100
            @warn "pepsgeneral did not converge within 100 iterations, conv=$conv"
            break
        end
    end
    return A
end

function rand_gauge(A)
    D = size(A, 1)
    atype = _arraytype(A)
    # U1, _ = Zygote.@ignore TeneT.qrpos(randn(ComplexF64, D, D))
    # U2, _ = Zygote.@ignore TeneT.qrpos(randn(ComplexF64, D, D))
    U1 = Zygote.@ignore atype(random_invertible_qr(D; T=ComplexF64))
    U2 = Zygote.@ignore atype(random_invertible_qr(D; T=ComplexF64))

    # @show U1[1]
    # @show U2[1]
    # return ein"(((abcdpn,ea),bf),cg),hd->efghpn"(A, U1, U2, inv(U1), inv(U2))
    return @tensor Aout[e,f,g,h,p,n] := A[a,b,c,d,p,n] * U1[e,a] * U2[b,f] * inv(U1)[c,g] * inv(U2)[h,d]
end

function random_invertible_qr(n::Int; T=Float64)
    # 生成随机正交矩阵Q和上三角矩阵R（对角线非零）
    Q = qr(rand(T, n, n)).Q
    R = triu(rand(T, n, n))
    
    # 确保R的对角线非零
    R[diagind(R)] .= rand(T, n) .+ 0.1  # 避免接近0的值
    
    return Matrix(Q * R)
end
