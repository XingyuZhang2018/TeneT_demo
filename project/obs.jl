using TeneT_demo
using CUDA
using Random

Random.seed!(100)
Ni, Nj  = 2, 2
model   = J1J2J3(Ni,Nj,1.0,0.3,0.1)
folder  = "./data/"
atype   = CuArray
D, χ    = 4, 80
tol     = 1e-10
maxiter = 50
miniter = 1


for targχ in 80:10:80
    observable(model, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
end