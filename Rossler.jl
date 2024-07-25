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

function force(t, p)
    if t <= 450
        return p
    else
        return p
    end
end

#Creating Rossler Data
u0 = [0.5, 0.1, 0.0]
omega = 2*pi*(1/10000)
function rossler(du, u, p, t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + 0.165*u[2]
    du[3] = 0.2 + (u[1]- 10 + sin(omega*t))*u[3]
end

tspan = (0, 700.0)
ross_prob = ODEProblem(rossler,u0,tspan)
sol = solve(ross_prob, Tsit5(), adaptive = false, dt = 0.1)

solu = sol[1:3,:]
t_p, tr_p, te_p, t_d, tr_d, te_d = data_generate(800, 4500, 7000, solu, sol.t)


#Creating Reservoir Parameters and predicted data
ResParams = ReservoirParams(6000, 0.001, 0.6, 1.0, 1.0, 0.0001)
data, _, _, _, _ = solve_system(ResParams, tr_d, te_d, 500, 1000, true, 2, te_d, false, 0, false, false, true)


#Plotting
true_data = solu[:,4500:6999]
predict_data = data[:,1:2500]

x = plot(te_p, true_data[1,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "x")
plot!(te_p, predict_data[1,:], label = "Predicted Data")
y = plot(te_p, true_data[2,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "y")
plot!(te_p, predict_data[2,:], label = "Predicted Data")
z = plot(te_p, true_data[3,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "z")
plot!(te_p, predict_data[3,:], label = "Predicted Data")
plot(x, y, z, layout = (3, 1))
savefig("RosslerPrediction.png")
