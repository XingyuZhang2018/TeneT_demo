module TeneT_demo

using FileIO
using KrylovKit
using LinearAlgebra
using LineSearches
using Random
using OptimKit
using OMEinsum
using Printf
using Parameters
using ForwardDiff
using Zygote
using TeneT
using CUDA

using TeneT: ALCtoAC, _arraytype, update!, mcform, rightenv, rightCenv
using TeneT: leg3, leg4
using TeneT: FLmap_parallel, FRmap_parallel, ACmap_parallel

import Base: Array
import CUDA: CuArray

export Heisenberg
export hamiltonian
export observable
export SUOptimize, FUOptimize, GradientOptimize, init_ipeps, optimise_ipeps, energy
abstract type iPEPSOptimize end

include("defaults.jl")
include("hamiltonian_models.jl")
include("precondition.jl")
include("optimise_patch.jl")
include("optimise_ipeps.jl")
include("init_ipeps_env.jl")
include("SU_parameterization.jl")
include("build_A_M.jl")
include("observable.jl")
include("SUFU.jl")

end
