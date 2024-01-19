# from probeai.data.make_dataset import load_probeai
# import torch
# import numpy as np
# import pytest


# @pytest.fixture
# def probeai_data():
#     return load_probeai()


# def test_data():
#     train, test = probeai_data()
#     assert len(train) == 108
#     assert len(test) == 92
#     assert len(set([train[i][0].shape for i in range(len(train))])) == 1
#     assert train[0][0].shape == torch.Size([1, 395, 70])
#     assert len(np.unique([train[i][1].numpy() for i in range(len(train))])) == 2
