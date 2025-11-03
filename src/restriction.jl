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
    R31 = ein"ab,cb->ac"(Rs[3], Rs[1])
    F = svd(R31)
    sqrtS = diagm(sqrt.(F.S))
    # @show F.S
    Rs3 = F.U * sqrtS
    Rs1 = conj(F.V * sqrtS)

    R42 = ein"ab,cb->ac"(Rs[4], Rs[2])
    F = svd(R42)
    # @show F.S
    sqrtS = diagm(sqrt.(F.S))
    Rs4 = F.U * sqrtS
    Rs2 = conj(F.V * sqrtS)

    return ein"(((abcdpn,dh),ae),bf),cg->efghpn"(A, Rs1,Rs2,Rs3,Rs4)
end

function ARstoA2(A, Rs)
    R3 = ein"ab,cb->ac"(Rs[3], Rs[1])
    R4 = ein"ab,cb->ac"(Rs[4], Rs[2])

    return ein"(abcdpn,bf),cg->afgdpn"(A, R3,R4)
end