# ReservoirComputer
Includes code for generating reservoir and prediction future data

The solve_system() function in "Predictor.jl" is the main function that's doing the prediction. 

"DataGeneration.jl" just contains a function that will split the given data into 3 phases: Transient Phase, Training Phase, and Testing Phase. It will split the data up into these 3 phases depending on what indices are passed onto the function.
"Lorenz.jl", "Rossler.jl", and "NeuronalModel.jl" all used the solve_system() function to predict behavior. 

Opening the Julia REPL and running any of these files with for example will produce plots predicting the behavior of the system.
As an example typing include("Rossler.jl") will produce a graph of the predicted data of the Rossler System. Reservoir parameters can be changed within these files. 

Running "VaryConnection.jl" will output the approximate Jacobian matrix, the true connection matrix, and the error between these matrices. 

"HyperparameterGraphing.jl" is only necessary when you want to see how the relative error changes with a changing reservoir parameter. 
