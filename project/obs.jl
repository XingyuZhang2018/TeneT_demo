using TeneT_demo
using CUDA
using Random

Random.seed!(100)
Ni, Nj  = 2, 2
model   = J1J2(Ni,Nj,1.0,0.5)
folder  = "./data/"
atype   = Array
D, χ    = 2, 10
tol     = 1e-10
maxiter = 10
miniter = 1


for targχ in 10:10:10
    observable(model, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
end