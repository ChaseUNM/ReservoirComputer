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



#########################################
##### Reservoir Code Below ##############
#########################################
mutable struct ReservoirParams
    Nodes::Int64
    Sparsity::Float64
    Rho::Float64
    alpha::Float64
    xi::Float64
    Ridge_param::Float64
end 

#Creates Adjacency Matrix
function Adjacency(nodes::Int, sparsity::Float64, rho::Float64, save_AW::Bool)
    #Create random, sparse Adjacency matrix
    A = sprand(nodes, nodes, sparsity)

    #Scale matrix so largest eigenvalue is rho
    l_eig = eigs(A, nev = 1)
    spectral_r = abs.(l_eig[1])[1]
    A = (A./spectral_r).*rho
    
    #save adjancency matrix if wanting to re-use
    if save_AW == true
        @save "A" A
    end
    return A
    
end



#Create random W_in matrix
function createW(nodes, train_data, save_AW::Bool)
    dim, steps = size(train_data)
    Win = zeros(nodes, dim)
    q = Integer(nodes/dim)
    #Each entry of W is drawn from Uniform distribution between -0.5 and 0.5  
    for i =1:dim
        Win[(i - 1)*q + 1: i*q, i] = rand(Uniform(-0.5, 0.5), (q, 1))
    end

    #save input weights if wanting to re-use
    if save_AW == true
        @save "Win" Win
    end
    return Win
end

#Create Q matrix, in this case the function we are applying to our nodes states is the identity function
function createQ(A, W, train_data, alpha, nodes, xi)
    dim, steps = size(train_data)
    #Depending on how data is generated will need to transpose
    R = zeros(nodes, steps)
    #R[:,1] = rand(nodes)
    for i=2:steps
        @inbounds R[:,i] = (1 - alpha)*R[:,i - 1] + alpha*tanh.(A*R[:,i - 1]+ W*train_data[:, i - 1] + xi*ones(nodes))
    end
    return R
end

#train on training set using ridge regression, \ was implmented in order to avoid using costly inv()
function train(train_data, Q, ridge_param)
    row, col = size(Q)
    x = Q[:,end]

    #This is squaring half of the nodes at each state, uncomment if not wanting to apply function to node states
    Q[2:2:row,:] = Q[2:2:row,:].^2
    
    storage = Matrix(1.0*I, row, row)
    mul!(storage, Q, Q', 1, ridge_param)

    F_in = storage'\(train_data*Q')'
    F_in = F_in'
    return F_in, x, Q
end

#To find the correct coefficients solve Q^T*F = X^T, this trains the model
#Next use this coefficient matrix F to predict future data
#This function outputs the predicted data
"""
function solve_system()
-------------
Arguments
-------------
ResParams: Reservoir Parameter
train_data: data used to train reservoir computer
test_data: data that will be used to test prediction of reservoir computer
P: retraining frequency
U: size of update window
Update: if True then update; if false then don't update
method: Update Method, 1 = Total Retraining, 2 = Moving Window Retraining
new_data: If updating need to constatnly input true data, this is that true data
new_F: if True then find output weights, if false then use a pre-selected output weight if wanting to re-use the same output weights for plotting or testing purposes
F: specific F you want to use if new_F is false
find_M: True if wanting to find approximate Jacobian
new_AW: True if wanting to create new adjacency and input weight matrix, false if wanting to use from file
save_AW: True if wanting to save adjancecy and input weight matrix, false if not wanting to save
--------------
outputs
--------------
predict_data: the reservoir computer prediction
F_in: the output weights after training the reservoir computer
M1: the approximate Jacobian matrix calculated before prediction
Mtot: an array of approximate Jacobians. This is only calculated if retraining and wanting to determine changing connections
residual_normal: the normalized training error
"""
function solve_system(ResParams::ReservoirParams, 
    train_data, test_data, P::Int64, U::Int64, update::Bool, 
    method::Int64, new_data, new_F::Bool, F, find_M::Bool, new_AW::Bool, save_AW::Bool)
    #Create A, W, and Q arrays
    nodes = ResParams.Nodes
    sparsity = ResParams.Sparsity
    rho = ResParams.Rho
    alpha = ResParams.alpha
    xi = ResParams.xi
    ridge_param = ResParams.Ridge_param

    if new_AW == true
        @time A = Adjacency(nodes, sparsity, rho, save_AW)
        @time W = createW(nodes, train_data, save_AW)
    else
        A = load_object("A")
        W = load_object("Win")
    end

    #Use stored adjacency and W_in matrices in order to reduce total time 


    if update == true
        if method == 2
            train_data = train_data[:,end - U + 1:end]
        end
    end
    @time Q = createQ(A, W, train_data, alpha, nodes, xi)
    #@save "Q_t" Q
    #println("Training...")   
    
    @time F_in, r, states = train(train_data, Q, ridge_param)

    if new_F == true
        F_in .= F 
    end
    #@save "F_in" F_in
    #println("Training complete")
    residual = norm(F_in*Q .- train_data)
    residual_normal = residual/norm(train_data)

    #println("Residual: ", residual)
    
    dim, test_steps = size(test_data)
    _, train_steps = size(train_data)
    
    #Create empty array to store predicted data
    predict_data = zeros(dim, test_steps)

    #Need last node states and data in order to start prediction

    x_in = zeros(dim)
    
    
    r_aug = zeros(size(r))
    #Run prediction
    M1 = zeros(dim, dim)
    Mtot = zeros(dim, dim, Int64(ceil(test_steps/P)))
    #Use counters to determine when to re-train parameters
    counter1 = 0
    counter2 = 0

    if train_steps < 1000
        av = train_steps - 1
    else
        av = 1000
    end
    if find_M == true
        println("Calculating approximate Jacobian")
        for i = train_steps - av:10:train_steps
            println("Train step: ", i)
            W_out_inv = pinv(F_in)
            W_in = W.*(1 .- tanh.(states[:,i]).^2)
            H = Matrix(A).*(1 .- tanh.(states[:,i]).^2)
            M1 .+= abs.((I - F_in*H*W_out_inv)\(F_in*W_in))
        end
    end
    M1 = M1./(av/10)
    println("Predicting...")
    for i = 1:test_steps
        counter2 += 1
        
        #This is squaring half of the nodes at each state, uncomment if not wanting to apply function to node states
        r_aug .= r
        r_aug[2:2:end] = r_aug[2:2:end].^2


        #Determine and append predicted data at next time step
        x_in = F_in*r_aug   

        predict_data[:, i] = x_in
                
        #println("Error in prediction for step ",i, ": ",  norm(test_data[:,i] - x_in))

        #Determine next node state  
        r .=  (1 - alpha)*r + alpha*tanh.(A*r + W*x_in + xi*ones(nodes))
        
        if update == true
            
            
            train_data = hcat(train_data, new_data[:,i])
            _, train_steps2 = size(train_data)
        
        #Retrain after every n steps
            if counter2 == P  

                println("Retraining...")


                #Method 1 does a total retrain, retrain on all the data
                if method == 1
                    Q = createQ(A, W, train_data, alpha, nodes, xi)
                    F, r, states = train(train_data, Q, ridge_param)
                    println("Difference between previous and new parameters: ", norm(F - F_in, 1))
                    F_in .= F

                #Method 2 has a moving window retrain, retrain using the previous U time steps
                elseif method == 2
                    train_data2 = train_data[:,train_steps2 - U + 1:train_steps2]
                    Q = createQ(A, W, train_data2, alpha, nodes, xi)
                    F, r, states = train(train_data2, Q, ridge_param)
                    println("Difference between previous and new parameters: ", norm(F - F_in, 1))
                    F_in .= F
                end
                counter1 += 1
                counter2 = 0
                M = zeros(dim, dim)
                
                if find_M == true
                    for j = train_steps - av:10: train_steps
                        println("Train step: ", j)
                        W_out_inv = pinv(F_in)
                        W_in = W.*(1 .- tanh.(states[:,j]).^2)
                        H = Matrix(A).*(1 .- tanh.(states[:,j]).^2)
                        M .+= abs.((I - F_in*H*W_out_inv)\(F_in*W_in))
                    end
                    M .= M./(av/10)
                    Mtot[:,:,counter1] = M 
                end
            end
            
        end

    end
    return predict_data, F_in, M1,Mtot, residual_normal
end
