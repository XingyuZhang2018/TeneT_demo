using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps, energy, optcont
using Optim
using OMEinsum

@testset "init_ipeps" for Ni = [1,2], Nj = [1,2], D in [2,3], χ in [10]
    model = Heisenberg(Ni,Nj)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ);
    @test size(A) == (D,D,D,D,2,Ni,Nj)
end

@testset "energy" for Ni = [1], Nj = [1], D in [2,3], χ in [10]
    model = Heisenberg(Ni,Nj)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ)
    oc = optcont(D, χ)
    h = hamiltonian(model)
    @show energy(h, A, oc, key; verbose = true, savefile = true)
end

@testset "precondition" begin
    χ = 10
    D = 2
    FLo = randn(χ,D,D,χ)
    FRo = randn(χ,D,D,χ)
    ACu = randn(χ,D,D,χ)
    ACd = randn(χ,D,D,χ)
    A = randn(D,D,D,D,2)
    Ap1 = ein"(((jafk,kbgl),abcde),mchl),jdim -> fghie"(FLo,ACd,A,FRo,ACu)
    ρ = ein"((jafk,kbgl),mchl),jdim -> afbgchdi"(FLo,ACd,FRo,ACu)
    Ap2 = ein"abcde,afbgchdi->fghie"(A, ρ)
    @test Ap1 ≈ Ap2 
end

@testset "optimise_ipeps" for Ni = [1], Nj = [1], D in [2], χ in [10]
    Random.seed!(100)
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= false)
    optimise_ipeps(A, key; ifprecondition = true,
        f_tol = 1e-10, opiter = 100, optimmethod = LBFGS(m = 20))
end

@testset "optimise_ipeps" for Ni = [2], Nj = [2], D in [2], χ in [10]
    model = Heisenberg(Ni,Nj,1.0,1.0,1.0)
    A, key = init_ipeps(model; Ni=Ni, Nj=Nj, D=D, χ=χ, verbose= false)
    optimise_ipeps(A, key; f_tol = 1e-6, opiter = 10, optimmethod = LBFGS(m = 20))
end