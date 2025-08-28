
"""
two-site contraction

```
                                            a ────┬──── c          
a ────┬──c ──┬──── f                        │     b     │  
│     b      e     │                        ├─ e ─┼─ f ─┤  
├─ g ─┼─  h ─┼─ i ─┤                        g     h     i 
│     k      n     │                        ├─ j ─┼─ k ─┤ 
j ────┴──l ──┴──── o                        │     m     │ 
                                            l ────┴──── n 
```
"""
# oc_H_leg3 = ein"(((agj,abc),gkhb),jkl),(((fio,cef),hnie),lno) -> "
# oc_V_leg3 = ein"(((abc,aeg),ehfb),cfi),(gjl,(jmkh,(ikn,lmn))) -> "
function oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; ifparallel, forloop_iter)
    l = FLmap_parallel(FLo, ACu, ACd, M1; ifparallel, forloop_iter)
    r = FRmap_parallel(FRo, ARu, ARd, M2; ifparallel, forloop_iter)
    return ein"abc,abc->"(l,r)
end

function oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; ifparallel, forloop_iter)
    u = ACmap_parallel(ACu, FLu, FRu, M1; ifparallel, forloop_iter)
    u = ACmap_parallel(u, FLo, FRo, M2; ifparallel, forloop_iter)
    return ein"abc,abc->"(u,ACd)
end

"""
two-site contraction

```
                                            a ────┬──── d          
a ────┬──d ──┬──── g                        │     bc    │  
│     bc     ef    │                        ├─ ef─┼─gh ─┤  u
├─ hi─┼─ jk ─┼─lm ─┤                        i     jk    l 
│     op     rs    │                        ├─ mn─┼─op ─┤  v
n ────┴──q ──┴──── t                        │     rs    │ 
      u      v                              q ────┴──── t 
```
"""
# oc_H_leg4 = ein"((((ahin,abcd),hojbu),ipkcu),nopq),((((glmt,defg),jrlev),ksmfv),qrst) -> "
# oc_V_leg4 = ein"((((abcd,aefi),ejgbu),fkhcu),dghl),(imnq,(mrojv,(nspkv,(lopt,qrst)))) -> "
function oc_H_leg4(FLo, ACu, A1u, A1d, ACd, FRo, ARu, A2u, A2d, ARd; ifparallel, forloop_iter)
    l = FLmap_parallel(FLo, ACu, ACd, A1u, A1d; ifparallel, forloop_iter)
    r = FRmap_parallel(FRo, ARu, ARd, A2u, A2d; ifparallel, forloop_iter)
    return ein"abcd,abcd->"(l,r)
end

function oc_V_leg4(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; ifparallel, forloop_iter)
    u = ACmap_parallel(ACu, FLu, FRu, A1u, A1d; ifparallel, forloop_iter)
    u = ACmap_parallel(u, FLo, FRo, A2u, A2d; ifparallel, forloop_iter)
    return ein"abcd,abcd->"(u,ACd)
end

function contract_n2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return sum(oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; ifparallel, forloop_iter))
end

function contract_o2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgchidm"(A1, O1, conj(A1)), D1^2,D2^2,D3^2*Dh,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afibgchdm"(A2, O2, conj(A2)), D1^2*Dh,D2^2,D3^2,D4^2)
    return sum(oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; ifparallel, forloop_iter))
end

function contract_n2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return sum(oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; ifparallel, forloop_iter))
end

function contract_o2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgichdm"(A1, O1, conj(A1)), D1^2,D2^2*Dh,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afbgchdmi"(A2, O2, conj(A2)), D1^2,D2^2,D3^2,D4^2*Dh)
    return sum(oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; ifparallel, forloop_iter))
end

function contract_n2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd; ifparallel, forloop_iter)
    return sum(oc_H_leg4(FLo, ACu, A1, conj(A1), ACd, FRo, ARu, A2, conj(A2), ARd; ifparallel, forloop_iter))
end

function contract_o2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; ifparallel, forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    A1u = reshape(ein"abcde,efi->abcidf"(A1, O1), D1,D2,D3*Dh,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->aibcdf"(A2, O2), D1*Dh,D2,D3,D4,d)
    return sum(oc_H_leg4(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; ifparallel, forloop_iter))
end

function contract_n2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd; ifparallel, forloop_iter)
    return sum(oc_V_leg4(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; ifparallel, forloop_iter))
end

function contract_o2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; ifparallel, forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    A1u = reshape(ein"abcde,efi->abicdf"(A1, O1), D1,D2*Dh,D3,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->abcdif"(A2, O2), D1,D2,D3,D4*Dh,d)
    return sum(oc_V_leg4(ACu, FLu, A1u, conj(A1), FRu, FLo, A2u, conj(A2), FRo, ACd; ifparallel, forloop_iter))
end

"""
one-site contraction

```                         
a ────┬──── c                      
│     b     │                      
├─ e ─┼─ f ─┤                      
│     h     │                      
g ────┴──── i                      
                            
```
"""
# oc1_leg3 = ein"(((aeg,abc),ehfb),ghi),cfi -> "
function oc1_leg3(FLo, ACu, M, ACd, FRo; ifparallel, forloop_iter)
    l = FLmap_parallel(FLo, ACu, ACd, M; ifparallel, forloop_iter)
    return ein"abc,abc->"(l,FRo)
end

"""
one-site contraction

```                         
a ────┬──── d                      
│     bc    │                      
├─ ef─┼─gh ─┤                      
│     jk    │                      
i ────┴──── l                      
                            
```
"""
# oc1_leg4 = ein"((((aefi,abcd),ejgbu),fkhcu),ijkl),dghl -> "
function oc1_leg4(FLo, ACu, Au, Ad, ACd, FRo; ifparallel, forloop_iter)
    l = FLmap_parallel(FLo, ACu, ACd, Au, Ad; ifparallel, forloop_iter)
    return ein"abcd,abcd->"(l,FRo)
end

function contract_n1(FLo::leg3, ACu, A, ACd, FRo; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return sum(oc1_leg3(FLo, ACu, M, ACd, FRo; ifparallel, forloop_iter))
end

function contract_o1(FLo::leg3, ACu, A, ACd, FRo, O; ifparallel, forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"(abcde,en),fghmn->afbgchdm"(A, O, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return sum(oc1_leg3(FLo, ACu, M, ACd, FRo; ifparallel, forloop_iter))
end

function contract_n1(FLo::leg4, ACu, A, ACd, FRo; ifparallel, forloop_iter)
    return sum(oc1_leg4(FLo, ACu, A, conj(A), ACd, FRo; ifparallel, forloop_iter))
end

function contract_o1(FLo::leg4, ACu, A, ACd, FRo, O; ifparallel, forloop_iter)
    return sum(oc1_leg4(FLo, ACu, ein"abcde,ef->abcdf"(A, O), conj(A), ACd, FRo; ifparallel, forloop_iter))
end

function expectation_value(A, env, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(A)
    atype = _arraytype(A[1])
    O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(params.model))
    etol = 0
    forloop_iter = params.forloop_iter
    ifparallel = params.boundary_alg.ifparallel
    len = length(A)
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; ifparallel, forloop_iter)
        n = contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; ifparallel, forloop_iter)
        n = contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); ifparallel, forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")

    return etol/len
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

function observable(A, χ, params::iPEPSOptimize; restriction_ipeps)
    D = size(A, 1)
    rt = initialize_vumps_runtime(A, D, χ, params; restriction_ipeps)

    A = restriction_ipeps(A)
    A = build_A(A, params)
    M = build_M(A, params)

    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    folder1 = joinpath(params.folder, "D$(D)", "VUMPS_rt_env")
    !(ispath(folder1)) && mkpath(folder1)
    params.ifsave_env && save_rt(folder1, rt; file="χ$(χ).jld2")
    env = VUMPSEnv(rt, M, params.boundary_alg)
    e = expectation_value(A, env, params)
    ξ = cor_len_value(env, params)
    return e, ξ
end
