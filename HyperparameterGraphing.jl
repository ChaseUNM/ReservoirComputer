#Hyperparameter Tuning
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

include("DataGeneration.jl")
include("Predictor.jl")
include("NeuronalModel.jl")

ResParams = ReservoirParams(6000, 0.001, 0.6, 1.0, 1.0, 0.0001)


#Each of the following graphs will plot the relative error for a changing Reservoir Parameter value. 

function rho_change(arr, samples)
    avg_arr = zeros(length(arr))
    err_arr = zeros(samples, length(arr))
    M_arr = zeros(6, 6, length(arr))
    for j = 1:samples
        for i = 1:length(arr)
            println("Sample $j for rho = ", arr[i])
            ResParams.Rho = arr[i]
            data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, true, false)
            tra_d = n_data[1,4000:4029]
            tra_p = te_p[1:30]
            x_data = data[1,1:30]
            M_arr[:, :, i] = M
            # println(tra_d)
            # println(x_data)
            rel_err = norm(x_data - tra_d)/norm(tra_d)
            println("Rel Err: $rel_err")
            avg_arr[i] += rel_err
            err_arr[j,i] = rel_err
        end
    end
    avg_arr .= avg_arr./samples
    std_dev = std(err_arr, dims = 1)'
    # std_dev = vec(std_dev)

    plot(arr, avg_arr, yerror = std_dev, left_margin = 8mm, bottom_margin = 8mm, xlabel = L"\rho", 
    label = L"\dfrac{||x - \hat{x}||}{||x||}", title = "rho vs. Relative Error", dpi = 250)
    savefig("RhoChange.png")

    @save "M_rho" M_arr

end



function beta_change(arr, samples)
    avg_arr = zeros(length(arr))
    err_arr = zeros(samples, length(arr))

    M_arr = zeros(6, 6, length(arr))
    for j = 1:samples
        for i = 1:length(arr)
            println("Sample $j for beta = ", arr[i])
            ResParams.Ridge_param = arr[i]
            if i == 1
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, true, true)
            else
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, false, false)
            end

            tra_d = n_data[1,4000:4029]
            tra_p = te_p[1:30]
            x_data = data[1,1:30]
            M_arr[:, :, i] = M
            # println(tra_d)
            # println(x_data)
            rel_err = norm(x_data - tra_d)/norm(tra_d)
            println("Rel Err: $rel_err")
            avg_arr[i] += rel_err
            err_arr[j,i] = rel_err
        end
    end
    avg_arr .= avg_arr./samples
    std_dev = std(err_arr, dims = 1)'
    std_dev = vec(std_dev)

    plot(arr, avg_arr, left_margin = 8mm,yerror = std_dev, bottom_margin = 8mm, xlabel = L"\beta", 
    label = L"\dfrac{||x - \hat{x}||}{||x||}", title = "beta vs. Relative Error", dpi = 250, yscale =:log10, xscale =:log10)
    savefig("BetaChange.png")

    @save "M_beta" M_arr
end

function alpha_change(arr, samples)
    avg_arr = zeros(length(arr))
    err_arr = zeros(samples, length(arr))
    M_arr = zeros(6, 6, length(arr))
    for j = 1:samples
        for i = 1:length(arr)
            ResParams.alpha = arr[i]
            println("Sample $j for alpha = ", arr[i])
            if i == 1
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, true, true)
            else   
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, false, false)
            end
            tra_d = n_data[1,4000:4029]
            tra_p = te_p[1:30]
            x_data = data[1,1:30]
            M_arr[:, :, i] = M
            # println(tra_d)
            # println(x_data)
            rel_err = norm(x_data - tra_d)/norm(tra_d)
            println("Rel Err: $rel_err")
            avg_arr[i] += rel_err
            err_arr[j,i] = rel_err
        end
    end
    avg_arr .= avg_arr./samples
    std_dev = std(err_arr, dims = 1)'
    std_dev = vec(std_dev)

    plot(arr, avg_arr, left_margin = 8mm,yerror = std_dev, bottom_margin = 8mm, xlabel = L"\alpha", 
    label = L"\dfrac{||x - \hat{x}||}{||x||}", title = "alpha vs. Relative Error", dpi = 250)
    savefig("alphaChange.png")

    @save "M_alpha" M_arr
end

function xi_change(arr, samples)
    avg_arr = zeros(length(arr))
    err_arr = zeros(samples, length(arr))
    M_arr = zeros(6, 6, length(arr))
    for j = 1:samples
        for i = 1:length(arr)
            println("Sample $j for xi = ", arr[i])
            ResParams.xi = arr[i]
            if i == 1
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, true, true)
            else
                data, _, M,_ , _= solve_system(ResParams, tr_d, te_d, 10, 1000, false, 1, te_d, false, 0, true, false, false)
            end
            tra_d = n_data[1,4000:4029]
            tra_p = te_p[1:30]
            x_data = data[1,1:30]
            M_arr[:, :, i] = M
            # println(tra_d)
            # println(x_data)
            rel_err = norm(x_data - tra_d)/norm(tra_d)
            println("Rel Err: $rel_err")
            avg_arr[i] += rel_err
            err_arr[j,i] = rel_err
        end
    end
    avg_arr .= avg_arr./samples
    std_dev = std(err_arr, dims = 1)'
    std_dev = vec(std_dev)

    plot(arr, avg_arr, left_margin = 8mm, yerror = std_dev,bottom_margin = 8mm, xlabel = L"\xi", 
    label = L"\dfrac{||x - \hat{x}||}{||x||}", title = "xi vs. Relative Error", dpi = 250)
    savefig("xiChange.png")

    @save "M_xi" M_arr
end

function window_change(arr, samples)
    avg_arr = zeros(length(arr))
    err_arr = zeros(samples, length(arr))
    M_arr = zeros(6, 6, length(arr))
    for j = 1:samples
        for i = 1:length(arr)
            println("Sample $j for window = ", arr[i])
            if i == 1
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 50, arr[i], true, 2, te_d, false, 0, true, true, true)
            else
                data, _, M, _, _= solve_system(ResParams, tr_d, te_d, 50, arr[i], true, 2, te_d, false, 0, true, false, false)
            end
            tra_d = n_data[1,4001:4030]
            tra_p = te_p[1:30]
            x_data = data[1,1:30]
            M_arr[:, :, i] = M
            # println(tra_d)
            # println(x_data)
            rel_err = norm(x_data - tra_d)/norm(tra_d)
            println("Rel Err: $rel_err")
            avg_arr[i] += rel_err
            err_arr[j,i] = rel_err
        end
    end
    avg_arr .= avg_arr./samples
    std_dev = std(err_arr, dims = 1)'
    std_dev = vec(std_dev)

    plot(arr, avg_arr, left_margin = 8mm,yerror = std_dev, bottom_margin = 8mm, xlabel = "Window Size", 
    label = L"\dfrac{||x - \hat{x}||}{||x||}", title = "Window Size vs. Relative Error", dpi = 250)
    savefig("WindowChange.png")
    @save "M_window" M_arr
end

sample = 1

function beta_arr(samples, start)
    n = zeros(samples)
    for i = 1:samples
        n[i] = start/(sqrt(10)^(i-1))
    end
    return n
end

rho_arr = LinRange(0.1, 0.9, 20)
alpha_arr = LinRange(0.2, 1.0, 20)
betas = beta_arr(20, 0.01)
xi_arr = LinRange(0.2, 1.0, 20)
window_arr = collect(199:200:3999)
rho_change(rho_arr, sample)
beta_change(betas, sample)
alpha_change(alpha_arr, sample)
xi_change(xi_arr, sample)
window_change(window_arr, sample)