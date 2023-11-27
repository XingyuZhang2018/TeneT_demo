using Test
using TeneT_demo
using TeneT_demo: HamiltonianModel

@testset "hamiltonian models" begin
    @test Ising(1,1,0.5) isa HamiltonianModel 
    @test Heisenberg(1,1,1.0,1.0,1.0) isa HamiltonianModel 
    @test Heisenberg(1,1) isa HamiltonianModel 
end

@testset "hamiltonian" begin
    h = hamiltonian(Heisenberg(1,1))
    @test size(h) == (2,2,2,2)
    rh = reshape(permutedims(h,(1,3,2,4)),4,4)
    @test rh' == rh
end

@testset "Heisenberg_bilayer" begin
    model = Heisenberg_bilayer(1,1)
    @test model isa HamiltonianModel

    H二, H⊥ = hamiltonian(model)
    H二 = reshape(H二, 16,16)
    @test H二' == H二'
    @test H⊥' == H⊥
    
    h1 = hamiltonian(model)
    h2 = hamiltonian_hand(model)
    @test h1[1] ≈ h2[1]
    @test h1[2] ≈ h2[2]
end