using FileIO
using LinearAlgebra: norm
using LineSearches
using Random
using Optim

export init_ipeps, optimise_ipeps

"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)
Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(model::HamiltonianModel; folder::String="./data/", atype = Array, Ni::Int, Nj::Int, D::Int, χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose = true)
    folder = joinpath(folder, "$(Ni)x$(Nj)/$(model)")
    mkpath(folder)
    chkp_file = joinpath(folder, "D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2")
    if isfile(chkp_file)
        A = load(chkp_file)["bcipeps"]
        verbose && println("load BCiPEPS from $chkp_file")
    else
        A = rand(ComplexF64,D,D,D,D,2,Ni,Nj)
        verbose && println("random initial BCiPEPS $chkp_file")
    end
    A /= norm(A)
    key = (folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose)
    return A, key
end

using OMEinsumContractionOrders

"""
    oc_H, oc_V = optcont(D::Int, χ::Int)
optimise the follow two einsum contractions for the given `D` and `χ` which are used to calculate the energy of the 2-site hamiltonian:
```
                                            a ────┬──── c 
a ────┬──c ──┬──── f                        │     b     │  
│     b      e     │                        ├─ e ─┼─ f ─┤  
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i 
│     k      n     │                        ├─ j ─┼─ k ─┤ 
j ────┴──l ──┴──── o                        │     m     │ 
                                            l ────┴──── n 
                                          
a ────┬──c     c──┬──── a                                 
│     b           b     │                                 
├─ d ─┼─ e     e ─┼─ d ─┤                                   
f     g           h     i 

f     g           h     i 
├─ d ─┼─ j     j ─┼─ d ─┤
│     b           b     │
a ────┴──k     k──┴──── a
                                          
```
where the central two block are six order tensor have extra bond `pq` and `rs`
"""
function optcont(D::Int, χ::Int)
    sd = Dict('a' => χ, 'b' => D^2,'c' => χ, 'e' => D^2, 'f' => χ, 'g' => D^2, 'h' => D^2, 'i' => D^2, 'j' => χ, 'k' => D^2, 'l' => χ, 'n' => D^2, 'o' => χ, 'p' => 2, 'q' => 2, 'r' => 2, 's' => 2)
    # for seed =20:100
    seed = 60
	Random.seed!(seed)
	# oc_H = optimize_code(ein"agj,abc,gkhbpq,jkl,fio,cef,hniers,lno -> pqrs", sd, TreeSA())
    oc_H = ein"(((agj,abc),gkhbpq),jkl),(((fio,cef),hniers),lno) -> pqrs"
	print("Horizontal Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(oc_H,sd),"\n")
    
    sd = Dict('a' => χ, 'b' => D^2, 'c' => χ, 'e' => D^2, 'f' => D^2, 'g' => χ, 'h' => D^2, 'i' => χ, 'j' => D^2, 'k' => D^2, 'l' => χ, 'm' => D^2, 'n' => χ, 'r' => 2, 's' => 2, 'p' => 2, 'q' => 2)
    # oc_V = optimize_code(ein"abc,aeg,ehfbpq,cfi,gjl,jmkhrs,ikn,lmn -> pqrs", sd, TreeSA())
    oc_V = ein"(((abc,aeg),ehfbpq),cfi),(gjl,(jmkhrs,(ikn,lmn))) -> pqrs"
    print("Vertical Contraction Complexity(seed=$(seed))",OMEinsum.timespace_complexity(oc_V,sd),"\n") 

    oc_corner = ein"(adf,abc),dgebpq->cefgpq", ein"(cba,adi),ehdbpq->cehipq",
                ein"(fda,abk),dbjgpq->fgjkpq", ein"(ida,kba),jbdhpq->hijkpq"
    oc_HV = ein"((cefgpq,cehist),fgjkuv),hijkwx->pqstuvwx"
    oc_H, oc_V, oc_corner, oc_HV
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, A, oc, key; verbose = true, savefile = true)
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    ap = ein"abcdeij,fghmnij->afbgchdmenij"(A, conj(A))
    ap = atype(reshape(ap, D^2, D^2, D^2, D^2, 2, 2, Ni, Nj))
    M = ein"abcdeeij->abcdij"(ap)

    env = obs_env(M; χ = χ, tol = tol, maxiter = maxiter, miniter = miniter, verbose = verbose, savefile = savefile, infolder = folder, outfolder = folder)
    e = expectation_value(h, ap, env, oc, key)
    return e
end

function expectation_value(h, ap, env, oc, key)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    oc_H, oc_V, oc_corner, oc_HV = oc
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)

    etol = 0
    for j = 1:Nj, i = 1:Ni
        verbose && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        lr = oc_H(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,jr],ARu[:,:,:,i,jr],ap[:,:,:,:,:,:,i,jr],ARd[:,:,:,ir,jr])
        e = Array(ein"pqrs, pqrs -> "(lr,atype(h[1])))[]
        @show e
        n =  Array(ein"pprr -> "(lr))[]
        verbose && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        lr = oc_V(ACu[:,:,:,i,j],FLu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],FRu[:,:,:,i,j],FL[:,:,:,ir,j],ap[:,:,:,:,:,:,ir,j],FR[:,:,:,ir,j],ACd[:,:,:,irr,j])
        e = Array(ein"pqrs, pqrs -> "(lr,atype(h[1])))[]
        n = Array(ein"pprr -> "(lr))[]
        verbose && println("Vertical energy = $(e/n)")
        etol += e/n

        jr  = mod1(j + 1, Nj)
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        lt = oc_corner[1](FLu[:,:,:,i,j], ACu[:,:,:,i,j], ap[:,:,:,:,:,:,i,j])
        rt = oc_corner[2](ARu[:,:,:,i,jr], FRu[:,:,:,i,jr], ap[:,:,:,:,:,:,i,jr])
        lb = oc_corner[3](FL[:,:,:,ir,j], ACd[:,:,:,irr,j], ap[:,:,:,:,:,:,ir,j])
        rb = oc_corner[4](FR[:,:,:,ir,jr], ARd[:,:,:,irr,jr], ap[:,:,:,:,:,:,ir,jr])
        
        lrtb = oc_HV(lt, rt, lb, rb)
        n = Array(ein"ppssuuww -> "(lrtb))[]
        e1 = Array(ein"pqssuuwx, pqwx -> "(lrtb,atype(h[2])))[]
        e2 = Array(ein"ppstuvww, stuv -> "(lrtb,atype(h[2])))[]
        verbose && println("J2_1 energy = $(e1/n)")
        verbose && println("J2_2 energy = $(e2/n)")
        etol += (e1 + e2)/n
    end

    verbose && println("e = $(etol/Ni/Nj)")
    return etol/Ni/Nj
end

"""
    optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))

return the tensor `A'` that describes an ipeps that minimises the energy of the
two-site hamiltonian `h`. The minimization is done using `Optim` with default-method
`LBFGS`. Alternative methods can be specified by loading `LineSearches` and
providing `optimmethod`. Other options to optim can be passed with `optimargs`.
The energy is calculated using vumps with key include parameters `χ`, `tol` and `maxiter`.
"""
function optimise_ipeps(A::AbstractArray, key; f_tol = 1e-6, opiter = 100, optimmethod = LBFGS(m = 20))
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key

    h = hamiltonian(model)
    oc = optcont(D, χ)
    f(x) = real(energy(h, x, oc, key))
    g(x) = Zygote.gradient(f,x)[1]
    message = "time  steps   energy           grad_norm\n"
    printstyled(message; bold=true, color=:red)
    flush(stdout)
    res = optimize(f, g, 
        A, optimmethod,inplace = false,
        Optim.Options(f_tol=f_tol, iterations=opiter,
        extended_trace=true,
        callback=os->writelog(os, key)),
        )
    return res
end

"""
    writelog(os::OptimizationState, key=nothing)

return the optimise infomation of each step, including `time` `iteration` `energy` and `g_norm`, saved in `/data/model_D_chi_tol_maxiter.log`. Save the final `ipeps` in file `/data/model_D_chi_tol_maxiter.jid2`
"""
function writelog(os::OptimizationState, key=nothing)
    message = "$(round(os.metadata["time"],digits=1))   $(os.iteration)       $(round(os.value,digits=10))    $(round(os.g_norm,digits=10))\n"

    printstyled(message; bold=true, color=:red)
    flush(stdout)

    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    !(isdir(folder)) && mkdir(folder)
    if !(key === nothing)
        logfile = open(joinpath(folder, "D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).log"), "a")
        write(logfile, message)
        close(logfile)
        save(joinpath(folder, "D$(D)_χ$(χ)_tol$(tol)_maxiter$(maxiter).jld2"), "bcipeps", os.metadata["x"])
    end
    return false
end