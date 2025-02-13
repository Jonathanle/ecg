"""
TODO: Create a entropy reducing infrastructure for pytorch code

1. Determine exactly what I want to be testing 
    - My subjective belief on importing tensors work
    - I can test by writing out the code by myself (
    - gradient knowledge computation
"""


import pytest 

from trainer import * # DF imprecise need better precision ono what i want 

import torch



@pytest.fixture(scope='module')
def patient_dataset():
    return 


def test_cuda_is_available():
    assert torch.cuda.is_available()



def test_tensor_is_cpu():
    """
    Test to Reduce the entropy of my understanding of what needs to be in a tensor

    Tensors will always be on a cpu
    # Surprise 1 - Tensor vs tensor i shoudl use tensor heuristically
    """
    tensor = torch.tensor([1, 2], dtype=torch.float32)

    assert tensor.device == torch.device('cpu')
    assert tensor.dtype == torch.float32

def test_tensor_is_in_cuda():
    """
    Test ability to get a tensor in devices 

    Surprise - torch.device('cuda') vs torch.device('cuda', index=0)
    indexes represent which "gpu" to compute something - df parallel processing
    
    torch.device('cuda') === torch.device('cuda:0') === torch.device('cuda', index=0)

    # in future - use cuda:0 ---> torch.device equates this with torch.device('cuda', index=0)

    """

    tensor = torch.tensor([1, 2], dtype=torch.float32).to(torch.device('cuda')) # cud

    # assumed = is equal to ==
    assert tensor.device == torch.device('cuda:0')


def test_loss_item():
    """
    Why is detach important?
    """


    tensor = torch.tensor([1], dtype=torch.int64)

    python_number = tensor.item()

    assert tensor.dtype == torch.int64
    assert isinstance(python_number, int)

def test_exception_loss_item():
    """
    Showing that i can test an exception occuring and then evaluate that to true + tensors .item only accepts tensors that are scalar

    This flow is conventional and arbritrary 
    """

    tensor = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(RuntimeError) as exc_info:
        python_number = tensor.item()

    assert str(exc_info.value) == "a Tensor with 2 elements cannot be converted to Scalar"

def test_CV_split():
    """
    Show that StratifiedKFold splits datasets into equal proportions of sample
    maybe these entropy can be temporararily corrected via using breakpoint() 
    """

    return
