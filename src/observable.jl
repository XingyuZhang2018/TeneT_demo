
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
function oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
    l = FLmap_forloop(FLo, ACu, ACd, M1; forloop_iter)
    r = FRmap_forloop(FRo, ARu, ARd, M2; forloop_iter)
    return ein"abc,abc->"(l,r)
end

function oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
    u = ACmap_forloop(ACu, FLu, FRu, M1; forloop_iter)
    u = ACmap_forloop(u, FLo, FRo, M2; forloop_iter)
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
function oc_H_leg4(FLo, ACu, A1u, A1d, ACd, FRo, ARu, A2u, A2d, ARd; forloop_iter)
    l = FLmap_forloop(FLo, ACu, ACd, (A1u, A1d); forloop_iter)
    r = FRmap_forloop(FRo, ARu, ARd, (A2u, A2d); forloop_iter)
    return sum(ein"abcd,abcd->"(l,r))
end

function oc_V_leg4(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; forloop_iter)
    u = ACmap_forloop(ACu, FLu, FRu, (A1u, A1d); forloop_iter)
    u = ACmap_forloop(u, FLo, FRo, (A2u, A2d); forloop_iter)
    return sum(ein"abcd,abcd->"(u,ACd))
end

function contract_n2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
end

function contract_o2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgchidm"(A1, O1, conj(A1)), D1^2,D2^2,D3^2*Dh,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afibgchdm"(A2, O2, conj(A2)), D1^2*Dh,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
end

function contract_n2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
end

function contract_o2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    M1 = reshape(ein"(abcde,eni),fghmn->afbgichdm"(A1, O1, conj(A1)), D1^2,D2^2*Dh,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    M2 = reshape(ein"(abcde,ien),fghmn->afbgchdmi"(A2, O2, conj(A2)), D1^2,D2^2,D3^2,D4^2*Dh)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
end

function contract_n2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter)
    return sum(oc_H_leg4(FLo, ACu, A1, conj(A1), ACd, FRo, ARu, A2, conj(A2), ARd; forloop_iter))
end

function contract_o2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    A1u = reshape(ein"abcde,efi->abcidf"(A1, O1), D1,D2,D3*Dh,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->aibcdf"(A2, O2), D1*Dh,D2,D3,D4,d)
    return oc_H_leg4(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; forloop_iter)
end

function contract_n2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter)
    return oc_V_leg4(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; forloop_iter)
end

function contract_o2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    A1u = reshape(ein"abcde,efi->abicdf"(A1, O1), D1,D2*Dh,D3,D4,d)
    D1,D2,D3,D4,d = size(A2)
    A2u = reshape(ein"abcde,ief->abcdif"(A2, O2), D1,D2,D3,D4*Dh,d)
    return oc_V_leg4(ACu, FLu, A1u, conj(A1), FRu, FLo, A2u, conj(A2), FRo, ACd; forloop_iter)
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
function oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter)
    l = FLmap_forloop(FLo, ACu, ACd, M; forloop_iter)
    return sum(ein"abc,abc->"(l,FRo))
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
function oc1_leg4(FLo, ACu, Au, Ad, ACd, FRo; forloop_iter)
    l = FLmap_forloop(FLo, ACu, ACd, (Au, Ad); forloop_iter)
    return sum(ein"abcd,abcd->"(l,FRo))
end

function contract_n1(FLo::leg3, ACu, A, ACd, FRo; forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter)
end

function contract_o1(FLo::leg3, ACu, A, ACd, FRo, O; forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    M = reshape(ein"(abcde,en),fghmn->afbgchdm"(A, O, conj(A)), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter)
end

function contract_n1(FLo::leg4, ACu, A, ACd, FRo; forloop_iter)
    return oc1_leg4(FLo, ACu, A, conj(A), ACd, FRo; forloop_iter)
end

function contract_o1(FLo::leg4, ACu, A, ACd, FRo, O; forloop_iter)
    return oc1_leg4(FLo, ACu, ein"abcde,ef->abcdf"(A, O), conj(A), ACd, FRo; forloop_iter)
end


"""
    next near neighbour contraction for 2 site
```
                       
a ────┬──c     c──┬──── a                                 
│     bl          bl    │                                 
├─ dm─┼─ eo    eo─┼─dm ─┤                                   
f     gn          hq    i 

f     gn          hq    i 
├─ dm─┼─ jr   jr ─┼─dm ─┤
│     bl          bl    │
a ────┴──k     k──┴──── a
```

"""
oc_ul(FLu::leg4, ACu, Au11, Ad11) = ein"((admf,ablc),dgebp),mnolp->fgnceo"(FLu, ACu, Au11, Ad11)
oc_ur(ARu::leg4, FRu, Au12, Ad12) = ein"((cbla,admi),ehdbp),oqmlp->ceohqi"(ARu, FRu, Au12, Ad12)
oc_dl(FLo::leg4, ACd, Au21, Ad21) = ein"((fdma,ablk),dbjgp),mlrnp->fgnjrk"(FLo, ACd, Au21, Ad21)
oc_dr(FRo::leg4, ARd, Au22, Ad22) = ein"((idma,kbla),jbdhp),rlmqp->hqijrk"(FRo, ARd, Au22, Ad22)
const leg6 = Union{<:AbstractArray{T, 6}, StructArray{<:Vector{<:AbstractArray{T, 6}}}} where T
oc_4_corner(ul::leg6,ur,dl,dr) = sum(ein"((fgnceo,ceohqi),fgnjrk),hqijrk->"(ul,ur,dl,dr))

function oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter)
    if forloop_iter == 1
        ul = oc_ul(FLu, ACu, Au11, Ad11)
        ur = oc_ur(ARu, FRu, Au12, Ad12)
        dl = oc_dl(FLo, ACd, Au21, Ad21)
        dr = oc_dr(FRo, ARd, Au22, Ad22)
        return oc_4_corner(ul,ur,dl,dr)
    else
        χ = size(FLu, 1)
        χ_loop = cld(χ, forloop_iter)
        χ_ranges = [range(1 + (i-1)*χ_loop, min(i*χ_loop, χ)) for i in 1:forloop_iter]
        cols = fill(:,ndims(FLu)-1)
        s = 0
        for range1 in χ_ranges, range2 in χ_ranges
            ul = oc_ul(FLu[cols..., range1], ACu, Au11, Ad11)
            ur = oc_ur(ARu, FRu[cols..., range2], Au12, Ad12)
            dl = oc_dl(FLo[range1, cols...], ACd, Au21, Ad21)
            dr = oc_dr(FRo[range2, cols...], ARd, Au22, Ad22)
            s += oc_4_corner(ul,ur,dl,dr)
        end
        return s
    end
end

function contract_n_D(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22; forloop_iter)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, conj(A11), A12, conj(A12), A21, conj(A21), A22, conj(A22); forloop_iter)
end

function contract_o_D1(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{ComplexF64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    Au11 = reshape(ein"abcde,eni->abcidn"(A11, O1), D1,D2,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    Au12 = reshape(ein"abcde,ni->anbicde"(A12, IDh), D1*Dh,D2*Dh,D3,D4,d)
    Ad12 = conj(A12)
    Au21 = A21
    Ad21 = conj(A21)
    D1,D2,D3,D4,_ = size(A22)
    Au22 = reshape(ein"abcde,ien->abcdin"(A22, O2), D1,D2,D3,D4*Dh,d)
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter)
end

function contract_o_D2(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{ComplexF64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    Au11 = reshape(ein"abcde,ni->abncide"(A11, IDh), D1,D2*Dh,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    Au12 = reshape(ein"abcde,eni->aibcdn"(A12, O1), D1*Dh,D2,D3,D4,d)
    Ad12 = conj(A12)
    D1,D2,D3,D4,_ = size(A21)
    Au21 = reshape(ein"abcde,ien->abcdin"(A21, O2), D1,D2,D3,D4*Dh,d)
    Ad21 = conj(A21)
    Au22 = A22
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter)
end

function expectation_value(model::Heisenberg, A, env, params::iPEPSOptimize)
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
        e = ifcheckpoint ? checkpoint(contract_o2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter) : contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter) : contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = ifcheckpoint ? checkpoint(contract_o2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter) : contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter) : contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")

    return etol/len
end

function expectation_value(model::J1J2, A, env, params::iPEPSOptimize)
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
        e = ifcheckpoint ? checkpoint(contract_o2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter) : contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter) : contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = ifcheckpoint ? checkpoint(contract_o2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter) : contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter) : contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n

        # O1, O2 = Zygote.@ignore atype.(hamiltonian_trunc(J1J2(1.0,0.5,false)))
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        e1 = ifcheckpoint ? checkpoint(contract_o_D1, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter) : contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        e2 = ifcheckpoint ? checkpoint(contract_o_D2, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter) : contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n_D, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter) : contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
        params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
        params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
        etol += J2 * (e1/n + e2/n)
    end

    params.verbosity >= 4 && println("energy = $(etol/len)")

    return etol/len
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
        e = ifcheckpoint ? checkpoint(contract_o2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter) : contract_o2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_H, FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter) : contract_n2_H(FLo[i,j], ACu[i,j], A[i,j], conj(ACd[ir,j]), FRo[i,jr], ARu[i,jr], A[i,jr], conj(ARd[ir,jr]); forloop_iter)
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += J1 * e/n

        ir  =  mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni) 
        e = ifcheckpoint ? checkpoint(contract_o2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter) : contract_o2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]), O1, O2; forloop_iter)
        n = ifcheckpoint ? checkpoint(contract_n2_V, ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter) : contract_n2_V(ACu[i,j], FLu[i,j], A[i,j], FRu[i,j], FLo[ir,j], A[ir,j], FRo[ir,j], conj(ACd[irr,j]); forloop_iter)
        params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        etol += J1 * e/n

        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        if i % 2 == 1 && j % 2 == 0
            n = ifcheckpoint ? checkpoint(contract_n_D, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter) : contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e1 = ifcheckpoint ? checkpoint(contract_o_D1, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter) : contract_o_D1(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D1 = $(J2*e1/n)")
            etol += J2 * e1/n
        elseif i % 2 == 0 && j % 2 == 1
            n = ifcheckpoint ? checkpoint(contract_n_D, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter) : contract_n_D(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr]; forloop_iter)
            e2 = ifcheckpoint ? checkpoint(contract_o_D2, FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter) : contract_o_D2(FLu[i,j], FLo[ir,j], ACu[i,j], conj(ACd[irr,j]), FRu[i,jr], FRo[ir,jr], ARu[i,jr], conj(ARd[irr,jr]), A[i,j], A[i,jr], A[ir,j], A[ir,jr], O1, O2; forloop_iter)
            params.verbosity >= 4 && println("h2D2 = $(J2*e2/n)")
            etol += J2 * e2/n
        end
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
