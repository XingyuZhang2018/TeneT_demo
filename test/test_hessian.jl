using ForwardDiff, KrylovKit, Zygote
using TensorOperations
using Test
using OMEinsum
using LinearAlgebra
using Random
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

using Zygote, OMEinsum, Random, Test, LinearAlgebra
@testset "double gradient" begin
    Random.seed!(1234)
    N = 10
    M = randn(Float64, N, N)  
    v1 = randn(Float64, N)
    v2 = randn(Float64, N)
    
    foo1(v1, v2) = real(v1' * M * v2)
    foo2(v1, v2) = real(ein"a,ab,b->"(v1, M, v2)[])
    foo3(v1, v2) = real(ein"(a,ab),b->"(v1, M, v2)[])
    foo4(v1, v2) = real(@tensoropt v1[1] * M[1, 2] * v2[2])

    @test foo1(v1, v2) ≈ foo2(v1, v2) ≈ foo3(v1, v2) ≈ foo4(v1, v2)

    @show gradient(x1 -> foo1(x1, v2), v1)[1]
    @show gradient(x1 -> foo4(x1, v2), v1)[1]
    # @show gradient(x2 -> dot(v1, gradient(x1 -> foo1(x1, x2), v1)[1]), v2)[1] # works
    # @show gradient(x2 -> dot(v1, gradient(x1 -> foo2(x1, x2), v1)[1]), v2)[1] # works
    # @show gradient(x2 -> dot(v1, gradient(x1 -> foo4(x1, x2), v1)[1]), v2)[1] # works
    # @show gradient(x2 -> dot(v1, gradient(x1 -> foo3(x1, x2), v1)[1]), v2)[1] # does not works
end
