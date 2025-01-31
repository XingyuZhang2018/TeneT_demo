module TeneT_demo

using FileIO
using KrylovKit
using LinearAlgebra
using LineSearches
using Random
using Optim
using OMEinsum
using Printf
using Parameters
using Zygote
using TeneT

using TeneT: ALCtoAC, _arraytype, update!, mcform, rightenv, rightCenv

export Heisenberg
export hamiltonian
export observable
export iPEPSOptimize, init_ipeps, optimise_ipeps, energy

include("defaults.jl")
include("hamiltonian_models.jl")
include("precondition.jl")
include("optimise_ipeps.jl")
include("init_ipeps.jl")
include("build_A_M.jl")
include("observable.jl")

end
