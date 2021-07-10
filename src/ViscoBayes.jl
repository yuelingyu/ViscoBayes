#!/usr/bin/env julia
__precompile__(true)
module ViscoBayes

export mcmcsample
export estimateMAP

using Random, Distributions, StatsBase, Statistics, Mamba
using LinearAlgebra, Interpolations, Optim, DataStructures
using Plots, Gadfly
using CSV, DataFrames, Dates
using ProgressMeter, BenchmarkTools

include("viscoelastic_model.jl")
include("statistic_model.jl")
include("pre_proc.jl")
include("mcmc_sampler.jl")
include("post_proc.jl")
include("stochastic_anlysis.jl")


end # module
