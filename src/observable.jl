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


function expectation_value(h, ap, env, oc, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(ap)
    oc_H, oc_V = oc
    etol = 0
    for p in 1:length(ap)
        i, j = Tuple(findfirst(==(p), ap.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        lr = oc_H(FLo[i,j],ACu[i,j],ap[i,j],conj(ACd[ir,j]),FRo[i,jr],ARu[i,jr],ap[i,jr],conj(ARd[ir,jr]))
        e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        n = Array(ein"pprr -> "(lr))[]
        params.verbosity >= 4 && println("Horizontal energy = $(e/n)")
        etol += 2*e/n

        # ir  =  mod1(i + 1, Ni)
        # irr = mod1(Ni - i, Ni) 
        # lr = oc_V(ACu[i,j],FLu[i,j],ap[i,j],FRu[i,j],FLo[ir,j],ap[ir,j],FRo[ir,j],conj(ACd[irr,j]))
        # e = Array(ein"pqrs, pqrs -> "(lr,h))[]
        # n = Array(ein"pprr -> "(lr))[]
        # params.verbosity >= 4 && println("Vertical energy = $(e/n)")
        # etol += e/n
    end

    params.verbosity >= 4 && println("energy = $(etol/length(ap))")
    return etol/length(ap)
end