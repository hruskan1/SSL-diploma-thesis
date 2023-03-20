import torch
import sys
sys.path.append("..")

import mixmatch as mixmatch

def test_shape_of_avg_predictions():
    """Test snippet of mixmatch averaging labels for classifications"""
    # Define input values
    aug_predictions = torch.zeros((6, 4, 5)) # N*K,C,..
    N = 3
    K = 2

    # Calculate avg_prediction
    avg_prediction = aug_predictions.reshape(N, K, *aug_predictions.shape[1:]).mean(dim=1, keepdim=True)

    # Verify shape of avg_prediction
    assert avg_prediction.shape == torch.Size([N, 1, *aug_predictions.shape[1:]])

    # Calculate avg_predictions
    avg_predictions = avg_prediction.repeat(1,K,*([1] * (aug_predictions.ndim-1))).reshape(N*K,*aug_predictions.shape[1:])

    # Verify shape of avg_predictions
    assert avg_predictions.shape == torch.Size([N*K, *aug_predictions.shape[1:]])

    avg_prediction = mixmatch.average_labels(aug_predictions.reshape(N,K, *aug_predictions.shape[1:]))
    print(avg_predictions.shape)

def test_sharpen():
    # Create example input tensor of shape [N, C, H, W]
    p = torch.nn.functional.softmax(torch.rand((2, 4)),dim=1)

    # Sharpen the tensor along the C dimension
    T = 0.3
    dim = 1
    p_sharp = mixmatch.sharpen(p, T, dim)

    assert p.shape == p_sharp.shape
    
    # Print shapes of input and output tensors
    # print(p)
    # print(p_sharp)

def test_mixup():
    # Create some fake data and labels
    data1 = torch.randn(10, 3, 32, 32)
    labels1 = torch.randn(10, 10)
    data2 = torch.randn(10, 3, 32, 32)
    labels2 = torch.randn(10, 10)

    # Create input tuples
    input_tuple1 = (data1, labels1)
    input_tuple2 = (data2, labels2)

    # Mix up the data and labels
    mixed_data, mixed_labels, lam = mixmatch.mixup(input_tuple1, input_tuple2)

    assert mixed_data.shape == data1.shape
    assert mixed_labels.shape == labels1.shape
    assert lam.shape == data1.shape[0:1]

    # print("Mixed-up data shape:", mixed_data.shape)
    # print("Mixed-up labels shape:", mixed_labels.shape)
    # print("Mixing coefficients shape:", lam.shape)
    
    

if __name__ == "__main__":
    #test_shape_of_avg_predictions()
    #test_sharpen()
    test_mixup()



