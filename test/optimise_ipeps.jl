using CUDA
using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: optcont
using Optim
using OMEinsum

@testset "init_ipeps" for Ni = [1,2], Nj = [1,2], D in [2,3]
    A = init_ipeps(; Ni, Nj, D=D);
    @test size(A) == (D,D,D,D,2,Ni,Nj)
end

@testset "energy" for Ni = [1], Nj = [1], D in [2,3], χ in [10]
    model = Heisenberg(Ni,Nj)
    h = hamiltonian(model)
    A = init_ipeps(; Ni, Nj, D=D)
    A = [A[:,:,:,:,:,i,j] for i = 1:size(A,6), j = 1:size(A,7)]
    oc = optcont(D, χ)
    boundary_alg = VUMPS()
    params = iPEPSOptimize(boundary_alg=boundary_alg, reuse_env = false, verbosity = 0)
    M = [reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D^2,D^2,D^2,D^2) for A in A]
    rt = VUMPSRuntime(M, χ, boundary_alg)
    @test energy(A, h, rt, oc, params) ≈ 0.5 atol=1e-1
end


@testset "optimise_ipeps $atype" for atype in [Array], Ni = [1], Nj = [1], D in [4], χ in [30]
    Random.seed!(100)
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    h = atype(hamiltonian(model))
    A = init_ipeps(;atype, Ni, Nj, D=D)
    boundary_alg = VUMPS(ifdownfromup=false, 
                         maxiter=10, 
                         miniter=1, 
                         verbosity=2
    )
    params = iPEPSOptimize(boundary_alg=boundary_alg, 
                           reuse_env=true, 
                           verbosity=3, 
                           maxiter=100,
                           tol=1e-10,
                           folder="data/$model/"
    )
    optimise_ipeps(A, h, χ, params)
end

@testset "optimise_ipeps $atype" for atype in [Array], Ni = [2], Nj = [2], D in [4], χ in [30]
    Random.seed!(100)
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    h = atype(hamiltonian(model))
    A = init_ipeps(;atype, Ni, Nj, D=D)
    boundary_alg = VUMPS(ifdownfromup=false, 
                         maxiter=10, 
                         miniter=1, 
                         verbosity=2
    )
    params = iPEPSOptimize(boundary_alg=boundary_alg, 
                           reuse_env=true, 
                           verbosity=3, 
                           maxiter=100,
                           tol=1e-10,
                           folder="data/$model/"
    )
    optimise_ipeps(A, h, χ, params)
end