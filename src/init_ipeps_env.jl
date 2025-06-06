"""
    init_ipeps(model::HamiltonianModel; D::Int, χ::Int, tol::Real, maxiter::Int)
Initial `bcipeps` and give `key` for use of later optimization. The key include `model`, `D`, `χ`, `tol` and `maxiter`. 
The iPEPS is random initial if there isn't any calculation before, otherwise will be load from file `/data/model_D_chi_tol_maxiter.jld2`
"""
function init_ipeps(;atype = Array, No, pattern, D::Int, d::Int, χ::Int, params)
    if No != 0
        file = "$(params.folder)/D$(D)_χ$(χ)/ipeps/ipeps_No.$(No).jld2"
        A = load(file, "bcipeps")
        @info "load ipeps from $file"
    else
        # Ni, Nj = size(pattern)
        A = rand(ComplexF64, D,D,D,D,d, length(unique(pattern)))
        # A = randSA(ComplexF64, atype, pattern, [(D,D,D,D,d) for i in 1:length(unique(pattern))])
        A /= norm(A)
        @info "random initial ipeps"
        # A = [A[:,:,:,:,:,i,j] for i = 1:Ni, j = 1:Nj]
        # A = restriction_ipeps(A)
        # M = [reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D^2,D^2,D^2,D^2) for A in A]
        # rt = VUMPSRuntime(M, χ, params.boundary_alg)
        # rt = leading_boundary(rt, M, params.boundary_alg)
        # env = VUMPSEnv(rt, M, params.boundary_alg)
        # @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
        # n, _ = rightenv(ARu, conj(ARd), M, FLo; ifobs=true, alg=params.boundary_alg) 
        # A /= sqrt(n[1])
        # A = reshape(A[1], D, D, D, D, d, Ni, Nj)
    end
    return atype(A)
end

function init_ipeps_from_small_D(;atype = Array, No, Ni::Int, Nj::Int, D::Int, D_new::Int, d::Int, χ::Int, ϵ::Real, params::iPEPSOptimize)
    file = "$(params.folder)/D$(D)_χ$(χ)/ipeps/ipeps_No.$(No).jld2"
    Aold = load(file, "bcipeps")
    println("load ipeps from $file")
    A = rand(ComplexF64, D_new, D_new, D_new, D_new, d, Ni, Nj) * ϵ
    A[1:D, 1:D, 1:D, 1:D, :, :, :] = Aold
    return atype(A)
end

function initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)
    # Construct the expected file path
    folder_path = joinpath(params.folder, "D$(D)_χ$(χ)")
    file_path = joinpath(folder_path, "VUMPS_rt_env.jld2")

    if params.ifload_env
        if ispath(file_path)
            try
                return load_rt(folder_path, _arraytype(A))
            catch e
                @warn "Failed to load runtime environment from $file_path: $(sprint(showerror, e)). Creating new environment."
                return create_new_runtime(A, χ, params; restriction_ipeps)
            end
        else
            @warn "File $file_path does not exist. Creating new VUMPS runtime environment."
            return create_new_runtime(A, χ, params; restriction_ipeps)
        end
    else
        return create_new_runtime(A, χ, params; restriction_ipeps)
    end
end

function create_new_runtime(A, χ, params; restriction_ipeps)
    A = restriction_ipeps(A)
    A = build_A(A, params)
    M = build_M(A, params)
    return VUMPSRuntime(M, χ, params.boundary_alg)
end
