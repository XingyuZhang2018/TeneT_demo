using TeneT_demo
using Random
using CUDA
using TeneT

Random.seed!(42)
atype = CuArray
D, χ = 4, 30
Ni, Nj = 1, 1
model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
h = atype(hamiltonian(model))
A = init_ipeps(;atype, Ni, Nj, D=D)
boundary_alg = VUMPS(ifupdown=false,
                     ifdownfromup=false, 
                     maxiter=10, 
                     miniter=1, 
                     verbosity=2
)
params = iPEPSOptimize(boundary_alg=boundary_alg, 
                       reuse_env=true, 
                       verbosity=4, 
                       maxiter=100,
                       tol=1e-10,
                       folder="data/$model/"
)
optimise_ipeps(A, h, χ, params)