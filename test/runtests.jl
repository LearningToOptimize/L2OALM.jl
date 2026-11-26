using Test
using Random
using JuMP
using Ipopt
using L2OCore
using L2OALM
using FiniteDifferences
using Zygote
using Lux
using Optimisers
using ExaModels
using KernelAbstractions
using BatchNLPKernels: BatchModel, BatchModelConfig

include("problems.jl")

@testset "L2OALM.jl" begin
    @testset "method and pipeline" begin
        include("test_method.jl")
    end
    @testset "properties of the method" begin
        include("test_properties.jl")
    end
    @testset "Lux training (Algorithm 1)" begin
        include("test_lux.jl")
    end
    @testset "BatchNLPKernels evaluator" begin
        include("test_batchnlp.jl")
    end
    @testset "code quality" begin
        include("test_quality.jl")
    end
end
