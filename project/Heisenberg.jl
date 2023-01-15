using ArgParse
using Random
using Test
using TeneT
using TeneT_demo
using TeneT_demo: init_ipeps
using Optim
using CUDA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--tol"
            help = "tol error for vumps"
            arg_type = Float64
            default = 1e-10
        "--maxiter"
            help = "max iterition for vumps"
            arg_type = Int
            default = 10
        "--miniter"
            help = "min iterition for vumps"
            arg_type = Int
            default = 1
        "--opiter"
            help = "iterition for optimise"
            arg_type = Int
            default = 200
        "--f_tol"
            help = "tol error for optimise"
            arg_type = Float64
            default = 1e-10
        "--Ni"
            help = "Cell size Ni"
            arg_type = Int
            required = true
        "--Nj"
            help = "Cell size Nj"
            arg_type = Int
            required = true
        "--D1"
            help = "ipeps virtual bond dimension"
            arg_type = Int
            required = true
        "--D2"
            help = "ipeps virtual bond dimension"
            arg_type = Int
            required = true
        "--chi"
            help = "vumps virtual bond dimension"
            arg_type = Int
            required = true
        "--folder"
            help = "folder for output"
            arg_type = String
            default = "./data/"
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    Random.seed!(100)
    Ni = parsed_args["Ni"]
    Nj = parsed_args["Nj"]
    D1 = parsed_args["D1"]
    D2 = parsed_args["D2"]
    χ = parsed_args["chi"]
    tol = parsed_args["tol"]
    maxiter = parsed_args["maxiter"]
    miniter = parsed_args["miniter"]
    opiter = parsed_args["opiter"]
    f_tol = parsed_args["f_tol"]
    folder = parsed_args["folder"]
    model = Heisenberg(Ni,Nj,-1.0,-1.0,1.0)
    A, key = init_ipeps(model; folder = folder, atype = CuArray, Ni=Ni, Nj=Nj, D=[D1,D2,D1,D2], χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose = true)
    optimise_ipeps(A, key; f_tol = f_tol, opiter = opiter)
end

main()