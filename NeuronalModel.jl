using LinearAlgebra
using Distributions
using Plots
using Measures
using JLD2
using LaTeXStrings
using StatsBase
using Arpack

using("Predictor.jl")

#slowA is the connection matrix between the x-variables in each node, commented lines are different cases of a changing variable
function slowA(n)
    A = zeros(6,6)
    A[1, 3] = 0.75
    A[3, 1] = 0.4
    A[1, 5] = 0.3
    A[5, 1] = 0.1
    # A[5, 1] = 0.2 + 0.1*tanh(n - 4001)
    # A[5, 1] = 0.1 + 0.2(sin(pi * n/4000))^2
    return A 
end

#Create neuronal model with only x connections
function neuronal_model(u0, nsteps, nodes)
    data = zeros(nodes*2, nsteps)
    data[:,1] = u0
    for j = 1:nodes
        for i = 2:nsteps
            A = slowA(i)
            data[2*j - 1,i] = data[2*j - 1, i - 1]^2*exp(data[2*j, i - 1] - data[2*j - 1, i - 1]) + 0.03 + dot(A[2*j - 1, :],data[:,i - 1])
            data[2*j, i] = 0.89 * data[2*j, i - 1] - 0.18*data[2*j - 1, i - 1] + 0.28 + dot(A[2*j, :],data[:,i - 1])
        end
    end
    return data
end

u0 = zeros(6)
nsteps = 6040
nodes = 3

n_data = neuronal_model(u0, nsteps, nodes)

t_p, tr_p, te_p, t_d, tr_d, te_d = data_generate(1, 6000, 6040, n_data, collect(1:1:6040))


#Uncomment if you want to plot the neuron dynamics

#Create predicted data
# ResParams = ReservoirParams(6000, 0.001, 0.6, 1.0, 1.0, 0.0001)
# data, _, _, _, _ = solve_system(ResParams, tr_d, te_d, 50, 100, true, 1, te_d, false, 0, false, true, false)

# true_data = n_data[:,6000:6039]
# predict_data = data[:,1:40]


# x = plot(te_p, true_data[1,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "Node 1")
# plot!(te_p, predict_data[1,:], label = "Predicted Data")
# y = plot(te_p, true_data[3,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "Node 2")
# plot!(te_p, predict_data[3,:], label = "Predicted Data")
# z = plot(te_p, true_data[5,:], label = "True Data", ls=:dash, bottom_margin = 8mm, left_margin = 8mm, dpi = 250, ylabel = "Node 3")
# plot!(te_p, predict_data[5,:], label = "Predicted Data")
# savefig("NeuronalPrediction.png")

