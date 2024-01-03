using FileIO
using LinearAlgebra: norm
using LineSearches
using Random
using Optim
using ADFPCM

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
    oc_H, oc_V
end

"""
    energy(h, bcipeps; χ, tol, maxiter)
return the energy of the `bcipeps` 2-site hamiltonian `h` and calculated via a
BCVUMPS with parameters `χ`, `tol` and `maxiter`.
"""
function energy(h, A, oc, key; verbose = true, savefile = true)
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    ap = ein"abcdeij,fghmnij->afbgchdmenij"(A, conj(A))
    ap = reshape(ap, D^2, D^2, D^2, D^2, 2, 2, Ni, Nj)
    M = ein"abcdeeij->abcdij"(ap)

    MArray = M[:,:,:,:,1,1]
	Random.seed!(42)
	env = FPCM(permutedims(MArray, (4,1,2,3)), 
    ADFPCM.Params(tol = 1e-10, χ=χ, 
                  maxiter=10, miniter=1, 
                  infolder = folder, verbose = verbose)
    )

    # env = obs_env(M; χ = χ, tol = tol, maxiter = maxiter, miniter = miniter, verbose = verbose, savefile = savefile, infolder = folder, outfolder = folder)
    e = expectation_value(h, ap, env, oc, key)
    return e
end

function expectation_value(h, ap, env, oc, key)
    # _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key
    oc_H, oc_V = oc
    # ACu = ALCtoAC(ALu, Cu)
    # ACd = ALCtoAC(ALd, Cd)

	@unpack Cul, Cld, Cdr, Cru, Tu, Tl, Td, Tr = env
	# save("./example/M.jld2", "M", MArray)
    
	ACu = ein"abc->cba"(Tu)
    ACd = Td
    ARu = ein"abc->cba"(Tu)
    ARd = Td
	FL = ein"(ab,bcd),de->ace"(Cul,Tl,Cld)
	FR = ein"(ab,bcd),de->eca"(Cdr,Tr,Cru)

	FLu = ein"ab,bcd->acd"(Cul,Tl)
	FRu = ein"abc,cd->dba"(Tr,Cru)
    FLd = ein"abc,cd->abd"(Tl,Cld)
    FRd = ein"ab,bcd->dca"(Cdr,Tr)

    etol = 0
    for j = 1:Nj, i = 1:Ni
        verbose && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = j + 1 - (j==Nj) * Nj
        lr = oc_H(FL,ACu,ap[:,:,:,:,:,:,i,j],ACd,FR,ARu,ap[:,:,:,:,:,:,i,jr],ARd)
        e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n =  Array(ein"pprr -> "(lr))[]
        verbose && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  i + 1 - (i==Ni) * Ni
        irr = Ni - i + (i==Ni) * Ni
        lr = oc_V(ACu,FLu,ap[:,:,:,:,:,:,i,j],FRu,FLd,ap[:,:,:,:,:,:,ir,j],FRd,ACd)
        e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n = Array(ein"pprr -> "(lr))[]
        verbose && println("Vertical energy = $(e/n)")
        etol += e/n
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