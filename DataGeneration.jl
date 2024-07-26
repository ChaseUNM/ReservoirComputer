#Data generation process

#Generates transient, training, testing data based on given data and indices. 
function data_generate(transient_idx, train_idx, test_idx, data, time_span)
    transient_points = vec(time_span[1: transient_idx])
    train_points = vec(time_span[transient_idx + 1: train_idx])
    test_points = vec(time_span[train_idx + 1: test_idx])

    transient_data = data[:,1:transient_idx]
    train_data = data[:,transient_idx + 1: train_idx]
    test_data = data[:,train_idx + 1: test_idx]


    return transient_points, train_points, test_points, transient_data, train_data, test_data
end
