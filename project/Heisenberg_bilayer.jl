using TeneT_demo

Ni,Nj = 2,2
D, χ  = 2,20
model = Heisenberg_bilayer(Ni,Nj,3.0,1.0)

A, key = init_ipeps(model; 
                    Ni=Ni, Nj=Nj, 
                    d=4, D=D, χ=χ, 
                    maxiter = 50,
                    verbose=true)

optimise_ipeps(A, key; 
               maxiter_ad = 10, miniter_ad = 3,
               f_tol = 1e-6, opiter = 100)
