using LinearAlgebra
using Distributions
using SparseArrays
using Plots
using NPZ
using Measures
using MAT
using JLD2
using OrdinaryDiffEq
using LaTeXStrings
using Statistics
using BlockDiagonals
using StatsBase
using Arpack

include("Predictor.jl")
include("DataGeneration.jl")


#Create Lorenz Data
u0 = [0.5, 0.1, 0.0]
omega = 0

function lorenz(du, u, p, t)
    du[1] = 10*(u[2] - u[1])
    du[2] = u[1] * (28 - u[3]) - u[2]
    du[3] = u[1]*u[2] - (8/3 + 0.2*sin(omega*t))*u[3]
end

tspan = (0.0, 3000.0)

lor_prob = ODEProblem(lorenz, u0, tspan)
sol = solve(lor_prob, Tsit5(), adaptive = false, dt = 0.1)
solu = sol[1:3,:]

t_p, tr_p, te_p, t_d, tr_d, te_d = data_generate(600, 5000, 5500, solu, sol.t)

#Create predicted data
ResParams = ReservoirParams(6000, 0.001, 0.6, 1.0, 1.0, 0.0001)

data, _, _, _, _ = solve_system(ResParams, tr_d, te_d, 50, 100, true, 1, te_d, false, 0, false, true, false)

#plotting data
true_data = solu[1,5000:5499]
predict_data = data[1,1:500]

x = plot(te_p, true_data[1,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "x")
plot!(te_p, predict_data[1,:], label = "Predicted Data")
y = plot(te_p, true_data[2,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "y")
plot!(te_p, predict_data[2,:], label = "Predicted Data")
z = plot(te_p, true_data[3,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "z")
plot!(te_p, predict_data[3,:], label = "Predicted Data")
plot(x, y, z, layout = (3, 1))
savefig("LorenzPrediction.png")
