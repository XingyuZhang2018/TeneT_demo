function build_A(A, params)
    Ar = StructArray([A[:,:,:,:,:,i] for i in 1:length(unique(params.pattern))], params.pattern)
    if params.ifSU
        return SU_parameterization(Ar, params; D_new=size(Ar[1],1))
    else
        return Ar
    end
end

function build_A(A, params, rt)
    A = StructArray(A, params.pattern)
    if params.SUτ != 0.0
        for i in 1:4
            A = one_bond_SU(A, params)
            A = map(x->permutedims(x, (2,3,4,1,5)), A)
        end
        
        A = hv_SU_update(A, params)
        # A = hv_FU_update(A, params, rt)
        return A
    else
        return A
    end
end

function build_M(A, params)
    D = size(A[1], 1)
    len = length(unique(params.pattern))
    if params.ifflatten
        # return StructArray([reshape(ein"abcde,fghme->afbgchdm"(A[i], conj(A[i])), D^2,D^2,D^2,D^2) for i in 1:len], params.pattern)
        return StructArray([begin
            @tensor M[a,f,b,g,c,h,d,m] := A[i][a,b,c,d,e] * conj(A[i][f,g,h,m,e])
            reshape(M, D^2,D^2,D^2,D^2)
        end for i in 1:len], params.pattern)
    else
        return A
    end
end