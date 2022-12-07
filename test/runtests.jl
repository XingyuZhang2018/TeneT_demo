using TeneT_demo
using Test

@testset "TeneT_demo.jl" begin
    include("hamiltonian_models.jl")
    include("contruct_M.jl")
    include("observable.jl")
    include("optimise_ipeps.jl")
end
