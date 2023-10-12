using OMEinsum
using TeneT: ALCtoAC

export observable

"""
    observable(env, model::MT, type)
return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, model::MT, ::Val{:Z}) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    atype = _arraytype(ALu)
    M   = atype(model_tensor(model, Val(:bulk)))
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    
    z_tol = 1
    ACu = ALCtoAC(ALu, Cu)

    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],FRu[:,:,:,i,j],conj(ACu[:,:,:,ir,j]))
        λ = ein"(acd,ab),(bce,de) ->"(FLu[:,:,:,i,jr],Cu[:,:,i,j],FRu[:,:,:,i,j],conj(Cu[:,:,ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return z_tol^(1/Ni/Nj)
end

function observable(env, model::MT, type) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    atype = _arraytype(ALu)
    M     = atype(model_tensor(model, Val(:bulk)))
    M_obs = atype(model_tensor(model, type      ))
    obs_tol = 0
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)

    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        obs = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M_obs[:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
        λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
        obs_tol += Array(obs)[]/Array(λ)[]
    end
    if type == Val(:mag)
        obs_tol = abs(obs_tol)
    end
    return obs_tol/Ni/Nj
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.

function observable(model, folder, atype, D, χ, targχ, tol, maxiter, miniter, Ni, Nj; ifload = false)
    A, key = init_ipeps(model; folder=folder, atype=atype, Ni=Ni, Nj=Nj, D=D, χ=χ, tol=tol,maxiter=maxiter, miniter=miniter, verbose = true)
    folder, model, atype, Ni, Nj, D, χ, tol, maxiter, miniter, verbose = key

    ap = ein"abcdeij,fghmnij->afbgchdmenij"(A, conj(A))
    ap = atype(reshape(ap, D^2, D^2, D^2, D^2, 2, 2, Ni, Nj))
    M = ein"abcdeeij->abcdij"(ap)
    a = atype(zeros(ComplexF64, D^2,D^2,D^2,D^2,Ni,Nj))
    for j in 1:Nj, i in 1:Ni
        a[:,:,:,:,i,j] = ein"ijklaa -> ijkl"(ap[:,:,:,:,:,:,i,j])
    end

    chkp_file_obs = folder*"obs_D$(D^2)_χ$(targχ).jld2"
    FL, FR = load(chkp_file_obs)["env"]
    chkp_file_up = folder*"up_D$(D^2)_χ$(targχ).jld2"                     
    rtup = SquareVUMPSRuntime(a, chkp_file_up, targχ; verbose = false)   
    FLu, FRu, ALu, ARu, Cu = rtup.FL, rtup.FR, rtup.AL, rtup.AR, rtup.C
    chkp_file_down = folder*"down_D$(D^2)_χ$(targχ).jld2"                              
    rtdown = SquareVUMPSRuntime(a, chkp_file_down, targχ; verbose = false)   
    ALd,ARd,Cd = rtdown.AL,rtdown.AR,rtdown.C
    ACu = ALCtoAC(ALu, Cu)
    ACd = ALCtoAC(ALd, Cd)

    ALu, Cu, ACu, ARu, ALd, Cd, ACd, ARd, FL, FR, FLu, FRu = map(atype, [ALu, Cu, ACu, ARu,ALd, Cd, ACd, ARd, FL, FR, FLu, FRu])

    M = zeros(ComplexF64, Ni, Nj, 3)
    for j = 1:Nj, i = 1:Ni
        ir = Ni + 1 - i
        lr = ein"(((aeg,abc),ehfbpq),ghi),cfi -> pq"(FL[:,:,:,i,j],ACu[:,:,:,i,j],ap[:,:,:,:,:,:,i,j],ACd[:,:,:,ir,j],FR[:,:,:,i,j])
        Mx = ein"pq, pq -> "(Array(lr),Sx)[]
        My = ein"pq, pq -> "(Array(lr),Sy)[]
        Mz = ein"pq, pq -> "(Array(lr),Sz)[]
        n = Array(ein"pp -> "(lr))[]
        M[i,j,1] = Mx/n
        M[i,j,2] = My/n
        M[i,j,3] = Mz/n
    end
    @show M
    M_n = zeros(ComplexF64, Ni, Nj)
    for j = 1:Nj, i = 1:Ni
        M_n[i,j] = norm(M[i,j,:])
    end
    @show M_n

end
