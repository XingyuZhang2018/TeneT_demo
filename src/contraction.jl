
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
function oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
    l = FLmap_forloop(FLo, ACu, ACd, M1; forloop_iter)
    r = FRmap_forloop(FRo, ARu, ARd, M2; forloop_iter)
    # return sum(ein"abc,abc->"(l,r))
    return @tensor l[a,b,c] * r[a,b,c]
end

function oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
    u = ACmap_forloop(ACu, FLu, FRu, M1; forloop_iter)
    u = ACmap_forloop(u, FLo, FRo, M2; forloop_iter)
    # return sum(ein"abc,abc->"(u,ACd))
    return @tensor u[a,b,c] * ACd[a,b,c]
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
    # return sum(ein"abcd,abcd->"(l,r))
    return @tensor l[a,b,c,d] * r[a,b,c,d]
end

function oc_V_leg4(ACu, FLu, A1u, A1d, FRu, FLo, A2u, A2d, FRo, ACd; forloop_iter)
    u = ACmap_forloop(ACu, FLu, FRu, (A1u, A1d); forloop_iter)
    u = ACmap_forloop(u, FLo, FRo, (A2u, A2d); forloop_iter)
    # return sum(ein"abcd,abcd->"(u,ACd))
    return @tensor u[a,b,c,d] * ACd[a,b,c,d]
end

function contract_n2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    # M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    M1 = reshape((@tensor out[a,f,b,g,c,h,d,m] := A1[a,b,c,d,e] * conj(A1[f,g,h,m,e])), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    # M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    M2 = reshape((@tensor out[a,f,b,g,c,h,d,m] := A2[a,b,c,d,e] * conj(A2[f,g,h,m,e])), D1^2,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
end

function contract_o2_H(FLo::leg3, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    # M1 = reshape(ein"(abcde,eni),fghmn->afbgchidm"(A1, O1, conj(A1)), D1^2,D2^2,D3^2*Dh,D4^2)
    M1 = reshape((@tensor out[a,f,b,g,c,h,i,d,m] := A1[a,b,c,d,e] * O1[e,n,i] * conj(A1[f,g,h,m,i])), D1^2,D2^2,D3^2*Dh,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    # M2 = reshape(ein"(abcde,ien),fghmn->afibgchdm"(A2, O2, conj(A2)), D1^2*Dh,D2^2,D3^2,D4^2)
    M2 = reshape((@tensor out[a,f,i,b,g,c,h,d,m] := A2[a,b,c,d,e] * O2[i,e,n] * conj(A2[f,g,h,m,n])), D1^2*Dh,D2^2,D3^2,D4^2)
    return oc_H_leg3(FLo, ACu, M1, ACd, FRo, ARu, M2, ARd; forloop_iter)
end

function contract_n2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    # M1 = reshape(ein"abcde,fghme->afbgchdm"(A1, conj(A1)), D1^2,D2^2,D3^2,D4^2)
    M1 = reshape((@tensor out[a,f,b,g,c,h,d,m] := A1[a,b,c,d,e] * conj(A1[f,g,h,m,e])), D1^2,D2^2,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    # M2 = reshape(ein"abcde,fghme->afbgchdm"(A2, conj(A2)), D1^2,D2^2,D3^2,D4^2)
    M2 = reshape((@tensor out[a,f,b,g,c,h,d,m] := A2[a,b,c,d,e] * conj(A2[f,g,h,m,e])), D1^2,D2^2,D3^2,D4^2)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
end

function contract_o2_V(ACu::leg3, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter)
    D1,D2,D3,D4,_ = size(A1)
    Dh = size(O1, 3)
    # M1 = reshape(ein"(abcde,eni),fghmn->afbgichdm"(A1, O1, conj(A1)), D1^2,D2^2*Dh,D3^2,D4^2)
    M1 = reshape((@tensor out[a,f,b,g,i,c,h,d,m] := A1[a,b,c,d,e] * O1[e,n,i] * conj(A1[f,g,h,m,i])), D1^2,D2^2*Dh,D3^2,D4^2)
    D1,D2,D3,D4,_ = size(A2)
    # M2 = reshape(ein"(abcde,ien),fghmn->afbgchdmi"(A2, O2, conj(A2)), D1^2,D2^2,D3^2,D4^2*Dh)
    M2 = reshape((@tensor out[a,f,b,g,c,h,d,m,i] := A2[a,b,c,d,e] * O2[i,e,n] * conj(A2[f,g,h,m,n])), D1^2,D2^2,D3^2,D4^2*Dh)
    return oc_V_leg3(ACu, FLu, M1, FRu, FLo, M2, FRo, ACd; forloop_iter)
end

function contract_n2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd; forloop_iter)
    return oc_H_leg4(FLo, ACu, A1, conj(A1), ACd, FRo, ARu, A2, conj(A2), ARd; forloop_iter)
end

function contract_o2_H(FLo::leg4, ACu, A1, ACd, FRo, ARu, A2, ARd, O1, O2; forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    # A1u = reshape(ein"abcde,efi->abcidf"(A1, O1), D1,D2,D3*Dh,D4,d)
    A1u = reshape((@tensor out[a,b,c,i,d,f] := A1[a,b,c,d,e] * O1[e,f,i]), D1,D2,D3*Dh,D4,d)
    D1,D2,D3,D4,d = size(A2)
    # A2u = reshape(ein"abcde,ief->aibcdf"(A2, O2), D1*Dh,D2,D3,D4,d)
    A2u = reshape((@tensor out[a,i,b,c,d,f] := A2[a,b,c,d,e] * O2[i,e,f]), D1*Dh,D2,D3,D4,d)
    return oc_H_leg4(FLo, ACu, A1u, conj(A1), ACd, FRo, ARu, A2u, conj(A2), ARd; forloop_iter)
end

function contract_n2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd; forloop_iter)
    return oc_V_leg4(ACu, FLu, A1, conj(A1), FRu, FLo, A2, conj(A2), FRo, ACd; forloop_iter)
end

function contract_o2_V(ACu::leg4, FLu, A1, FRu, FLo, A2, FRo, ACd, O1, O2; forloop_iter)
    D1,D2,D3,D4,d = size(A1)
    Dh = size(O1, 3)
    # A1u = reshape(ein"abcde,efi->abicdf"(A1, O1), D1,D2*Dh,D3,D4,d)
    A1u = reshape((@tensor out[a,b,i,c,d,f] := A1[a,b,c,d,e] * O1[e,f,i]), D1,D2*Dh,D3,D4,d)
    D1,D2,D3,D4,d = size(A2)
    # A2u = reshape(ein"abcde,ief->abcdif"(A2, O2), D1,D2,D3,D4*Dh,d)
    A2u = reshape((@tensor out[a,b,c,d,i,f] := A2[a,b,c,d,e] * O2[i,e,f]), D1,D2,D3,D4*Dh,d)
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
    # return sum(ein"abc,abc->"(l,FRo))
    return @tensor l[a,b,c] * FRo[a,b,c]
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
    # return sum(ein"abcd,abcd->"(l,FRo))
    return @tensor l[a,b,c,d] * FRo[a,b,c,d]
end

function contract_n1(FLo::leg3, ACu, A, ACd, FRo; forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    # M = reshape(ein"abcde,fghme->afbgchdm"(A, conj(A)), D1^2,D2^2,D3^2,D4^2)
    M = reshape((@tensor out[a,f,b,g,c,h,d,m] := A[a,b,c,d,e] * conj(A[f,g,h,m,e])), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter)
end

function contract_o1(FLo::leg3, ACu, A, ACd, FRo, O; forloop_iter)
    D1,D2,D3,D4,_ = size(A)
    # M = reshape(ein"(abcde,en),fghmn->afbgchdm"(A, O, conj(A)), D1^2,D2^2,D3^2,D4^2)
    M = reshape((@tensor out[a,f,b,g,c,h,d,m] := A[a,b,c,d,e] * O[e,n] * conj(A[f,g,h,m,n])), D1^2,D2^2,D3^2,D4^2)
    return oc1_leg3(FLo, ACu, M, ACd, FRo; forloop_iter)
end

function contract_n1(FLo::leg4, ACu, A, ACd, FRo; forloop_iter)
    return oc1_leg4(FLo, ACu, A, conj(A), ACd, FRo; forloop_iter)
end

function contract_o1(FLo::leg4, ACu, A, ACd, FRo, O; forloop_iter)
    # return oc1_leg4(FLo, ACu, ein"abcde,ef->abcdf"(A, O), conj(A), ACd, FRo; forloop_iter)
    @tensor AO[a,b,c,d,f] := A[a,b,c,d,e] * O[e,f]
    return oc1_leg4(FLo, ACu, AO, conj(A), ACd, FRo; forloop_iter)
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
# oc_ul(FLu::leg4, ACu, Au11, Ad11) = ein"((admf,ablc),dgebp),mnolp->fgnceo"(FLu, ACu, Au11, Ad11)
# oc_ur(ARu::leg4, FRu, Au12, Ad12) = ein"((cbla,admi),ehdbp),oqmlp->ceohqi"(ARu, FRu, Au12, Ad12)
# oc_dl(FLo::leg4, ACd, Au21, Ad21) = ein"((fdma,ablk),dbjgp),mlrnp->fgnjrk"(FLo, ACd, Au21, Ad21)
# oc_dr(FRo::leg4, ARd, Au22, Ad22) = ein"((idma,kbla),jbdhp),rlmqp->jrkhqi"(FRo, ARd, Au22, Ad22)
oc_ul(FLu::leg4, ACu, Au11, Ad11) = @tensor out[f,g,n,c,e,o] := FLu[a,d,m,f] * ACu[a,b,l,c] * Au11[d,g,e,b,p] * Ad11[m,n,o,l,p]
oc_ur(ARu::leg4, FRu, Au12, Ad12) = @tensor out[c,e,o,h,q,i] := ARu[c,b,l,a] * FRu[a,d,m,i] * Au12[e,h,d,b,p] * Ad12[o,q,l,m,p]
oc_dl(FLo::leg4, ACd, Au21, Ad21) = @tensor out[f,g,n,j,r,k] := FLo[f,d,m,a] * ACd[a,b,l,k] * Au21[d,b,j,g,p] * Ad21[m,l,r,n,p]
oc_dr(FRo::leg4, ARd, Au22, Ad22) = @tensor out[j,r,k,h,q,i] := FRo[i,d,m,a] * ARd[k,b,l,a] * Au22[j,b,d,h,p] * Ad22[r,l,m,q,p]
const leg6 = Union{<:AbstractArray{T, 6}, StructArray{<:Vector{<:AbstractArray{T, 6}}}} where T
# oc_4_corner(ul::leg6,ur,dl,dr) = sum(ein"(fgnceo,ceohqi),(fgnjrk,jrkhqi)->"(ul,ur,dl,dr))
oc_4_corner(ul::leg6,ur,dl,dr) = @tensoropt ul[f,g,n,c,e,o] * ur[c,e,o,h,q,i] * dl[f,g,n,j,r,k] * dr[j,r,k,h,q,i]
oc_4_corner(FLu, ACu, Au11, Ad11,
            ARu, FRu, Au12, Ad12,
            FLo, ACd, Au21, Ad21,
            FRo, ARd, Au22, Ad22) = oc_4_corner(oc_ul(FLu, ACu, Au11, Ad11),
                                                oc_ur(ARu, FRu, Au12, Ad12),
                                                oc_dl(FLo, ACd, Au21, Ad21),
                                                oc_dr(FRo, ARd, Au22, Ad22))

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
        cols = fill(:, ndims(FLu)-1)
        s = 0.0
        for range1 in χ_ranges, range2 in χ_ranges
            s += checkpoint(oc_4_corner, FLu[cols..., range1], ACu, Au11, Ad11,
                                         ARu, FRu[cols..., range2], Au12, Ad12,
                                         FLo[range1, cols...], ACd, Au21, Ad21,
                                         FRo[range2, cols...], ARd, Au22, Ad22)

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
    # Au11 = reshape(ein"abcde,eni->abcidn"(A11, O1), D1,D2,D3*Dh,D4,d)
    Au11 = reshape((@tensor out[a,b,c,i,d,n] := A11[a,b,c,d,e] * O1[e,n,i]), D1,D2,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    # Au12 = reshape(ein"abcde,ni->anbicde"(A12, IDh), D1*Dh,D2*Dh,D3,D4,d)
    Au12 = reshape((@tensor out[a,n,b,i,c,d,e] := A12[a,b,c,d,e] * IDh[n,i]), D1*Dh,D2*Dh,D3,D4,d)
    Ad12 = conj(A12)
    Au21 = A21
    Ad21 = conj(A21)
    D1,D2,D3,D4,_ = size(A22)
    # Au22 = reshape(ein"abcde,ien->abcdin"(A22, O2), D1,D2,D3,D4*Dh,d)
    Au22 = reshape((@tensor out[a,b,c,d,i,n] := A22[a,b,c,d,e] * O2[i,e,n]), D1,D2,D3,D4*Dh,d)
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter)
end

function contract_o_D2(FLu::leg4, FLo, ACu, ACd, FRu, FRo, ARu, ARd, A11, A12, A21, A22, O1, O2; forloop_iter)
    Dh = size(O1, 3)
    atype = _arraytype(O1)
    IDh = Zygote.@ignore atype(Matrix{ComplexF64}(I, Dh, Dh))
    D1,D2,D3,D4,d = size(A11)
    # Au11 = reshape(ein"abcde,ni->abncide"(A11, IDh), D1,D2*Dh,D3*Dh,D4,d)
    Au11 = reshape((@tensor out[a,b,n,c,i,d,e] := A11[a,b,c,d,e] * IDh[n,i]), D1,D2*Dh,D3*Dh,D4,d)
    Ad11 = conj(A11)
    D1,D2,D3,D4,_ = size(A12)
    # Au12 = reshape(ein"abcde,eni->aibcdn"(A12, O1), D1*Dh,D2,D3,D4,d)
    Au12 = reshape((@tensor out[a,i,b,c,d,n] := A12[a,b,c,d,e] * O1[e,n,i]), D1*Dh,D2,D3,D4,d)
    Ad12 = conj(A12)
    D1,D2,D3,D4,_ = size(A21)
    # Au21 = reshape(ein"abcde,ien->abcdin"(A21, O2), D1,D2,D3,D4*Dh,d)
    Au21 = reshape((@tensor out[a,b,c,d,i,n] := A21[a,b,c,d,e] * O2[i,e,n]), D1,D2,D3,D4*Dh,d)
    Ad21 = conj(A21)
    Au22 = A22
    Ad22 = conj(A22)
    return oc_D_leg4(FLu, FLo, ACu, ACd, FRu, FRo, ARu, ARd, Au11, Ad11, Au12, Ad12, Au21, Ad21, Au22, Ad22; forloop_iter)
end
