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

local_gauge_contraction(A, G) = @tensor out[e,f,g,h,p] := A[a,b,c,d,p] * G[1][e,a] * G[2][b,f] * G[3][c,g] * G[4][h,d]

function guage_transfer(A, G, params)
    Gh, Gv = G
    pattern = params.pattern
    Ni, Nj = size(pattern)
    A′ = Zygote.Buffer(A)
    for q in 1:size(A, 6)
        i, j = Tuple(findfirst(==(q), pattern))
        ir = mod1(i - 1, Ni)
        jr = mod1(j - 1, Nj) 
        # G1 = exp(-(Gh[:,:,pattern[i,jr]] + Gh[:,:,pattern[i,jr]]')/2)
        # G2 = exp((Gv[:,:,q] + Gv[:,:,q]')/2)
        # G3 = exp((Gh[:,:,q] + Gh[:,:,q]')/2)
        # G4 = exp(-(Gv[:,:,pattern[ir,j]] + Gv[:,:,pattern[ir,j]]')/2)
        # A′[:,:,:,:,:,q] = local_gauge_contraction(A[:,:,:,:,:,q], [G1, G2, G3, G4])
        A′[:,:,:,:,:,q] = local_gauge_contraction(A[:,:,:,:,:,q], [inv(Gh[:,:,pattern[i,jr]]), Gv[:,:,q], Gh[:,:,q], inv(Gv[:,:,pattern[ir,j]])])
    end
    return copy(A′)
end

function find_local_min_norm_G(A, params)
    atype = _arraytype(A)
    A_cpu = Zygote.@ignore Array(A) 

    function f(G) 
        A′ = guage_transfer(A_cpu, G, params)
        return norm(A′)
    end
    function f2(G)
        A′ = guage_transfer(A_cpu, G, params)
        @tensor Ml[1,6] := A′[1,2,3,4,5,7] * conj(A′[6,2,3,4,5,7])
        @tensor Mr[6,3] := A′[1,2,3,4,5,7] * conj(A′[1,2,6,4,5,7])

        @tensor Mu[4,6] := A′[1,2,3,4,5,7] * conj(A′[1,2,3,6,5,7])
        @tensor Md[6,2] := A′[1,2,3,4,5,7] * conj(A′[1,6,3,4,5,7])

        return norm(Ml - Mr) + norm(Mu - Md)
    end

    function fg(G)
        cost, vjp = pullback(f, G)
        g = vjp(1)[1]
        return cost, g
    end 

    D, N = size(A)[[1, 6]]
    eltypeA = eltype(A)
    Gh = randn(eltypeA,D,D,N)
    Gv = randn(eltypeA,D,D,N)
    for q in 1:N
        Gh[:,:,q] = I(D)
        Gv[:,:,q] = I(D)
    end
    Ginit = [Gh, Gv]
    @info "initial norm = $(f(Ginit))"

    G, f, _ = optimize(fg, Ginit, LBFGS(maxiter=1000, gradtol=1e-15))
    @info "final norm = $f f2(G) = $(f2(G))"

    return atype.(G)
end

function local_min_norm(A, params; ifignore_gauge=true)
    if ifignore_gauge
        G = Zygote.@ignore find_local_min_norm_G(A, params)
    else
        G = find_local_min_norm_G(A, params)
    end
    AG = guage_transfer(A, G, params)

    return AG
end

function ChainRulesCore.rrule(::typeof(find_local_min_norm_G) , A, params)
    G = find_local_min_norm_G(A, params)
    atype = _arraytype(A)
    function find_local_min_norm_G_pullback(ΔG)
        ΔG = Array.(ΔG)
        A = Array(A)
        G = Array.(G)
        function fixpoint(A, G)
            AG = guage_transfer(A, G, params)

            # for q in 1:size(A, 6)
            #     i, j = Tuple(findfirst(==(q), pattern))
            #     ir = mod1(i - 1, Ni)
            #     jr = mod1(j - 1, Nj) 
                @tensor Ml[1,6] := AG[1,2,3,4,5,7] * conj(AG[6,2,3,4,5,7])
                @tensor Mr[6,3] := AG[1,2,3,4,5,7] * conj(AG[1,2,6,4,5,7])

                @tensor Mu[4,6] := AG[1,2,3,4,5,7] * conj(AG[1,2,3,6,5,7])
                @tensor Md[6,2] := AG[1,2,3,4,5,7] * conj(AG[1,6,3,4,5,7])

            return [Ml-Mr, Mu-Md]
        end

        cost, vjp = pullback(fixpoint, A, G)
        sum(norm.(cost)) >= 1e-10 && @warn "local min norm gauge condition not satisfied, cost=$cost"
        vjp_A(x) = vjp(x)[1]
        vjp_G(x) = vjp(x)[2]

        dA, info = linsolve(vjp_G, -ΔG; maxiter=1)
        if info == 0
            @warn "linesolve did not converge in find_local_min_norm_G_pullback, info=$info"
        end
        @show norm(dA)

        return NoTangent(), atype(vjp_A(dA)), NoTangent()
    end
    return G, find_local_min_norm_G_pullback
end

function find_local_hermite_G(A, params)
    atype = _arraytype(A)
    A_cpu = Zygote.@ignore Array(A) 

    function f(G) 
        A′ = guage_transfer(A_cpu, G, params)
        return norm(A′ - permutedims(A′, (3,2,1,4,5,6))) + norm(A′ - permutedims(A′, (1,4,3,2,5,6)))
    end
    function fg(G)
        cost, vjp = pullback(f, G)
        g = vjp(1)[1]
        return cost, g
    end 

    D, N = size(A)[[1, 6]]
    eltypeA = eltype(A)
    Gh = randn(eltypeA,D,D,N)
    Gv = randn(eltypeA,D,D,N)
    for q in 1:N
        Gh[:,:,q] = I(D)
        Gv[:,:,q] = I(D)
    end
    Ginit = [Gh, Gv]
    @info "initial norm = $(f(Ginit))"

    G, f, _ = optimize(fg, Ginit, LBFGS(maxiter=100))
    @info "final norm = $f"

    return atype.(G)
end

function local_hermite(A, params)
    G = Zygote.@ignore find_local_hermite_G(A, params)
    AG = guage_transfer(A, G, params)

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

"""
    to_mcf_ipeps(T; max_iter=1000, tol=1e-10)

将 iPEPS 张量 T 变换为最小正则形式 (MCF)。
输入张量 T 的索引顺序必须为 (l, d, r, u, p):
1. l: Left (Virtual)
2. d: Down (Virtual)
3. r: Right (Virtual)
4. u: Up (Virtual)
5. p: Physical
"""
function to_mcf_ipeps(T; max_iter=1000, tol=1e-12)
    # 获取维度
    # D_hor: 水平虚维度, D_ver: 垂直虚维度, dp: 物理维度
    Dl, Dd, Dr, Du, dp = size(T)
    @assert Dl == Dr "Horizontal virtual dimensions must be equal for MCF."
    @assert Dd == Du "Vertical virtual dimensions must be equal for MCF."
    
    curr_T = copy(T)
    
    for iter in 1:max_iter
        max_diff = 0.0
        
        # --- 1. 水平方向处理 (l & r) ---
        # 提取 Left (index 1) 的简约密度矩阵 rho_in
        # reshape 为 (Dl, Dd*Dr*Du*dp)
        ML = reshape(curr_T, Dl, :)
        rho_l = ML * ML'
        
        # 提取 Right (index 3) 的简约密度矩阵 rho_out
        # 先将索引 3 换到第 1 位: (r, l, d, u, p)
        MR = reshape(permutedims(curr_T, (3, 1, 2, 4, 5)), Dr, :)
        rho_r = MR * MR'
        
        # MCF 条件: rho_l = rho_r^T
        target_r = transpose(rho_r)
        diff_h = norm(rho_l - target_r) / norm(rho_l)
        
        # 计算并应用水平规范变换 g_h
        g_h = solve_balancing_gauge(rho_l, target_r)
        inv_gh_t = transpose(inv(g_h))
        
        # 更新 l (索引 1): T' = g_h * T
        curr_T = reshape(g_h * ML, Dl, Dd, Dr, Du, dp)
        
        # 更新 r (索引 3): T'' = T' * inv_gh_t (作用在索引 3 上)
        tmp_r = reshape(permutedims(curr_T, (3, 1, 2, 4, 5)), Dr, :)
        curr_T = permutedims(reshape(inv_gh_t * tmp_r, Dr, Dl, Dd, Du, dp), (2, 3, 1, 4, 5))
        

        # --- 2. 垂直方向处理 (d & u) ---
        # 提取 Down (index 2) 的简约密度矩阵 rho_in
        MD = reshape(permutedims(curr_T, (2, 1, 3, 4, 5)), Dd, :)
        rho_d = MD * MD'
        
        # 提取 Up (index 4) 的简约密度矩阵 rho_out
        MU = reshape(permutedims(curr_T, (4, 1, 2, 3, 5)), Du, :)
        rho_u = MU * MU'
        
        # MCF 条件: rho_d = rho_u^T
        target_u = transpose(rho_u)
        diff_v = norm(rho_d - target_u) / norm(rho_d)
        
        # 计算并应用垂直规范变换 g_v
        g_v = solve_balancing_gauge(rho_d, target_u)
        inv_gv_t = transpose(inv(g_v))
        
        # 更新 d (索引 2)
        tmp_d = reshape(permutedims(curr_T, (2, 1, 3, 4, 5)), Dd, :)
        curr_T = permutedims(reshape(g_v * tmp_d, Dd, Dl, Dr, Du, dp), (2, 1, 3, 4, 5))
        
        # 更新 u (索引 4)
        tmp_u = reshape(permutedims(curr_T, (4, 1, 2, 3, 5)), Du, :)
        curr_T = permutedims(reshape(inv_gv_t * tmp_u, Du, Dl, Dd, Dr, dp), (2, 3, 4, 1, 5))
        
        max_diff = max(diff_h, diff_v)
        if max_diff < tol
            @info "MCF converged in $iter iterations."
            return curr_T
        end
    end
    
    @warn "MCF did not fully converge. Final diff: $max_diff"
    return curr_T
end

"""
    solve_balancing_gauge(A, B)
求解 HAH = B 得到平衡变换矩阵 g = sqrt(H)。
这能够最小化张量的 Frobenius 范数。
"""
function solve_balancing_gauge(A, B; reg=1e-15)
    # 正则化以处理接近奇异的情况
    A_safe = A + reg * I
    B_safe = B + reg * I
    
    sqrtA = sqrt(Hermitian(A_safe))
    isqrtA = inv(sqrtA)
    # 核心公式: H = A^-1/2 * (A^1/2 * B * A^1/2)^1/2 * A^-1/2
    H = isqrtA * sqrt(Hermitian(sqrtA * B_safe * sqrtA)) * isqrtA
    return sqrt(Hermitian(H))
end

function local_min_norm_iter(A, params)
    T = A[:,:,:,:,:,1]
    AG = to_mcf_ipeps(T; max_iter=1000, tol=1e-10)


    @tensor Ml[1,6] := AG[1,2,3,4,5] * conj(AG[6,2,3,4,5])
    @tensor Mr[6,3] := AG[1,2,3,4,5] * conj(AG[1,2,6,4,5])

    @tensor Mu[4,6] := AG[1,2,3,4,5] * conj(AG[1,2,3,6,5])
    @tensor Md[6,2] := AG[1,2,3,4,5] * conj(AG[1,6,3,4,5])

    @show norm(Ml - Mr) + norm(Mu - Md)
    return reshape(AG, size(A))
end