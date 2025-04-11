using ForwardDiff, KrylovKit, Zygote
using Test

@testset "test_hessian.jl" begin
    function f(V)
        return sum(sin.(V) .+ V.^3)
    end

    compute_gradient(f, V) = Zygote.gradient(f, V)[1]

    # 计算Hessian-向量乘积
    function hessian_vec_prod(f, V, v)
        return ForwardDiff.derivative(t -> Zygote.gradient(f, V + t * v)[1], 0.0)
    end

    n = 10 
    V = randn(n)  
    g = compute_gradient(f, V)
    g_prime, info = linsolve(v->hessian_vec_prod(f, V, v), g; maxiter=1)
    H = ForwardDiff.hessian(f, V)
    g_prime_direct = H \ g
    @test g_prime ≈ g_prime_direct
end
