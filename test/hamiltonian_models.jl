using Test
using TeneT_demo
using TeneT_demo: HamiltonianModel

@testset "hamiltonian models" begin
    @test Heisenberg(1,1,-1.0,-1.0,1.0) isa HamiltonianModel 
end

@testset "hamiltonian" begin
    h1, h2 = hamiltonian(Heisenberg(1,1,-1.0,-1.0,1.0))
    @show size(h1)
    # @test size(h) == (2,2,2,2)
    # rh = reshape(permutedims(h,(1,3,2,4)),4,4)
    # @test rh' == rh
end