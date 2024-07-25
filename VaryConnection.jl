#Varying connection matrix, import data
using JLD2
using Plots
using Measures
using LaTeXStrings


include("Predictor.jl")

#Create Neuronal Model in order plot, can use this to slowly vary connections
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

A_conn = [0 0.75 0.3; 0.4 0 0; 0.1 0 0]

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

#Create predicted data, return the approximate Jacobian Matrix M
ResParams = ReservoirParams(6000, 0.001, 0.6, 1.0, 1.0, 0.0001)
data, _, M, _, _ = solve_system(ResParams, tr_d, te_d, 50, 100, true, 1, te_d, false, 0, true, true, false)

#Function changes 6 x 6 matrix to 3 x 3 matrix, used when we know we only have connections between the x variables. Will have
#to change function if more nodes are added or if connections aren't just between x-variables
function changeM(M)
    M2 = zeros(3, 3)
    M2[1,2] = M[1, 3]
    M2[1,3] = M[1, 5]
    M2[2,1] = M[3, 1]
    M2[2,3] = M[3, 5]
    M2[3,1] = M[5, 1]
    M2[3,2] = M[5, 3]
    return M2
end

#Normalizes inferred matrix relative to true matrix
function NormalizationM1(True, Infer)
    TrueS = sum(True)
    InferS = sum(Infer)
    n = InferS/TrueS
    InferN = Infer./n
    return InferN 
end

#Calculate error between true and inferred matrix using error metric in paper
function err_comp(True, Infer)
    sumA = sum(True.^2)
    row, col = size(True)
    E = 0
    for i = 1:row
        for j = 1:row
            E += (True[i,j] - Infer[i,j])^2
        end
    end 
    E = E/(row*(row - 1))
    E = E/sumA
    return E
end

M = changeM(M)
M = NormalizationM1(M, A_conn)
conn_diff = abs.(M - A_conn)


#Plotting heatmaps
h_diff = heatmap(conn_diff, c=:jet, clims = (0, 1), plot_title = "Error", dpi = 250, left_margin = 7mm, bottom_margin = 7mm, ticks = false,
yflip = true)
for i in 1:size(A_conn2, 1)
    for j in 1:size(A_conn2, 2)
        annotate!(j, i, text(string(round(conn_diff[i,j]; digits = 3)), :white, :center))
    end
end
h_og = heatmap(A_conn, c=:jet, clims = (0, 1), title = "True", top_margin = 5mm, ticks = false, yflip = true, dpi = 250)
for i in 1:size(A_conn2, 1)
    for j in 1:size(A_conn2, 2)
        annotate!(j, i, text(string(round(A_conn2[i,j]; digits = 3)), :white, :center))
    end
end
h_m = heatmap(M, c=:jet, clims = (0, 1), title = "Inferred", top_margin = 5mm, ticks = false, yflip = true, dpi = 250)
for i in 1:size(A_conn2, 1)
    for j in 1:size(A_conn2, 2)
        annotate!(j, i, text(string(round(M[i,j]; digits = 3)), :white, :center))
    end
end

plot(h_og, h_m, layout = (1, 2))
savefig("NormalizedConnectivity.png")
savefig(h_diff, "Difference.png")




function beta_arr(samples, start)
    n = zeros(samples)
    for i = 1:samples
        n[i] = start/(sqrt(10)^(i-1))
    end
    return n
end


A_conn = [0 0.75 0.3; 0.4 0 0; 0.1 0 0]
A_conn_2 = [0 0.75 0.3; 0.4 0 0; 0.3 0 0]

M_alpha = load_object("M_alpha")
M_beta = load_object("M_beta")
M_rho = load_object("M_rho")
M_xi = load_object("M_xi")
M_window = load_object("M_window")

rho_arr = LinRange(0.1, 0.9, 20)
alpha_arr = LinRange(0.2, 1.0, 20)
betas = beta_arr(20, 0.01)
xi_arr = LinRange(0.2, 1.0, 20)
window_arr = collect(199:200:3999)



conn_err_plot(A_conn, M_window)




# Merr_plot(M_arr, err_arr, A_conn2)

# println(err_comp(A_conn2, M0))
# println(err_comp(A_conn3, M_2000))
# println(err_comp(A_conn2, M))
# println(err_comp(A_conn2, M2))
# println(err_comp(A_conn2, M3))


# h1 = heatmap(M1, c=:jet, title = "Inferred at 4200", top_margin = 5mm)
# h2 = heatmap(M2, c=:jet, title = "Inferred at 4400 ", top_margin = 5mm)
# h3 = heatmap(M3, c=:jet, title = "Inferred at 4600", top_margin = 5mm)
# h4 = heatmap(M4, c=:jet, title = "Inferred at 4800", top_margin = 5mm)
# h5 = heatmap(M5, c=:jet, title = "Inferred at 5000", top_margin = 5mm)

# plot(h1, h5, layout = (1, 2), dpi = 250, left_margin = 8mm, bottom_margin = 8mm)
# savefig("HeatmapsVary.png")