
function expectation_value(model::Heisenberg, A, env, fδEiEI, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
    etol = 0
    forloop_iter = params.forloop_iter
    ifcheckpoint = params.ifcheckpoint
    len = length(A)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    Zygote.@ignore fδEiEI[4] = imag(etol/len)

    return etol/len
end

function expectation_value(model::J1J2, A, env, fδEiEI, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter, ifcheckpoint, bondratio, order = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
    etol = 0
    len = length(A)
    e_dict = Dict{String, Dict{String, Any}}(
        "Horizontal_energy" => Dict{String, Any}(),
        "Vertical_energy"   => Dict{String, Any}(),
        "Diagonal1_energy"  => Dict{String, Any}(),
        "Diagonal2_energy"  => Dict{String, Any}()
    )
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        J1h, J1v = enlarge_coupling(model, order, i, j, bondratio)

        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1h * e/n
        e_dict["Horizontal_energy"]["$(i),$(j)"] = J1h * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1v * e/n
        e_dict["Vertical_energy"]["$(i),$(j)"] = J1v * e/n

        if model.ifrotate
            O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(J1J2(model.J1,model.J2,false)))
        end
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        e2 = contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
        params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
        etol += J2 * (e1/n + e2/n)
        e_dict["Diagonal1_energy"]["$(i),$(j)"] = J2 * e1/n
        e_dict["Diagonal2_energy"]["$(i),$(j)"] = J2 * e2/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")
    Zygote.@ignore fδEiEI[4] = abs(imag(etol/len))
    return etol/len, e_dict
end

function expectation_value(model::SS, A, env, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J1, J2 = model
    @unpack forloop_iter, ifcheckpoint = params

    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(model))
    etol = 0
    len = length(A)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr], O1, O2; forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j], O1, O2; forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j]; forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n

        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        if i % 2 == 1 && j % 2 == 0
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e1 = contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
            etol += J2 * e1/n
        elseif i % 2 == 0 && j % 2 == 1
            n = contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e2 = contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
            etol += J2 * e2/n
        end
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")

    return etol/len
end

function magnetization_value(model, A, env, params)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))

    Ni, Nj = size(ACu)
    len = length(ACu.data)
    Ni,Nj = size(ACu)
    forloop_iter = params.forloop_iter
    m_dict = Dict{String, Any}()
    Mnorm = zeros(ComplexF64, Ni, Nj)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        Mx = contract_o1(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], Sx; forloop_iter)
        My = contract_o1(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], Sy; forloop_iter)
        Mz = contract_o1(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j], Sz; forloop_iter)
        
        n = contract_n1(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,j]; forloop_iter)
        Mag = [Mx/n, My/n, Mz/n]
        Mnorm[i,j] = norm(Mag)
        params.verbosity >= 4 && println("M = $(Mag)\n|M| = $(Mnorm)")

        m_dict["$(i),$(j)"] = Dict("Mx" => Mag[1], "My" => Mag[2], "Mz" => Mag[3], "|M|" => Mnorm[i,j])
    end

    M_mean = sum(Mnorm)/len
    params.verbosity >= 4 && println("|M|_mean = $(M_mean)")
    return M_mean, m_dict
end

function Cmap(C, Aui, Adi, J::Int)
    Nj = size(Aui, 1)
    for j = 1:Nj
        jr = mod1(J+j-1, Nj)
        C = TeneT.ρmap(C,Aui[jr],Adi[jr])
    end
    return C
end

function cor_len_value(env, params) 
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    # Ni = size(ACu, 1)
    # ξs = zeros(eltype(A[1]),Ni)
    Cint = TeneT.cellones(ACu)[1]
    # for i in 1:Ni
    #     # λcs, _, info = eigsolve(C->Cmap(C,ARu[i,:],conj(ARd[i,:]),1), Cint, 1, :LM; ishermitian = false)
    #     λcs, _, info = eigsolve(ρ->TeneT.ρmap(ρ,ARu[i,:],1), Cint, 2, :LM; maxiter=100, ishermitian = false)
    #     info.converged == 0 && @warn "cor_len not converged"
    #     @show λcs λcs[2]/λcs[1]
    #     ξs[i] = -1/log(abs(λcs[2]/λcs[1]))
    #     params.verbosity >= 4 && println("ξ = $(ξs[i])")
    # end
    # for i in 1:Ni
        λcs, _, info = eigsolve(C->Cmap(C,ARu[1,:],conj(ARu[1,:]),1), Cint, 10, :LM; maxiter=100, ishermitian = false)
        # λcs, _, info = eigsolve(ρ->TeneT.ρmap(ρ,ARu[i,:],1), Cint, 2, :LM; ishermitian = false)
        info.converged == 0 && @warn "cor_len not converged"
        @show λcs λcs[2]/λcs[1]
        λ2 = 0
        for i in 2:length(λcs)
            if !(norm(λcs[i]) ≈ norm(λcs[1]))
                λ2 = λcs[i]
                break
            end
        end
        
        ξ = -1/log(abs(λ2/λcs[1]))
        @show ξ
        params.verbosity >= 4 && println("ξ = $(ξ)")
    # end
    return ξ
end

function observable(A, χ, fδEiEI, params::iPEPSOptimize; restriction_ipeps = _restriction_ipeps)
    D = size(A, 1)
    rt = initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)

    A = restriction_ipeps(A)
    A = build_A(A, params)
    M = build_M(A, params)

    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    params.ifsave_env && save_rt(joinpath(params.folder, "D$(D)", "VUMPS_rt_env"), rt; file="χ$(χ).jld2")
    env = VUMPSEnv(rt, M, params.boundary_alg)
    e = expectation_value(params.model, A, env, fδEiEI, params)
    mag = magnetization_value(params.model, A, env, params)
    ξ = cor_len_value(env, params)
    write_obs_log(e, mag, ξ, χ, joinpath(params.folder, "D$(D)"), params)
    return e, mag, ξ
end

function write_obs_log(e, mag, ξ, χ, folder, ::iPEPSOptimize)
    path = joinpath(folder, "observable")
    isdir(path) || mkpath(path)
    obs_log = joinpath(path, "χ$χ.log")
    e_dict = e[2]
    m_dict = mag[2]

    open(obs_log, "w") do io
        write(io, @sprintf("energy_per_site:\n%.15f\n", real(e[1])))

        for bond_type in keys(e_dict)
            write(io, "$bond_type: i j energy\n")
            for pos in keys(e_dict[bond_type])
                write(io, @sprintf("%s %.15f\t", pos, real(e_dict[bond_type][pos])))
            end
            write(io, "\n")
        end
        
        write(io, @sprintf("magnetization_norm_per_site:\n%.15f\n", real(mag[1])))

        write(io, "magnetization: i j |M| Mx My Mz\n")
        for pos in keys(m_dict)
            write(io, @sprintf("%s %.15f %.15f %.15f %.15f\t", pos, real(m_dict[pos]["|M|"]), real(m_dict[pos]["Mx"]), real(m_dict[pos]["My"]), real(m_dict[pos]["Mz"])))
            write(io, "\n")
        end
        write(io, @sprintf("correlation_length:\n%.15f\n", real(ξ)))
    end
end
