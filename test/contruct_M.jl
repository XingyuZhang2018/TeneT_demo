using Test
using TeneT_demo

@testset "Ising model_tensor" for Ni in [1,2], Nj in [1,2]
    model = Ising(Ni,Nj,0.5)
    @test size(model_tensor(model, Val(:bulk))) == (2,2,2,2,Ni,Nj)
    @test size(model_tensor(model, Val(:mag) )) == (2,2,2,2,Ni,Nj)
    @test size(model_tensor(model, Val(:mag) )) == (2,2,2,2,Ni,Nj)
end