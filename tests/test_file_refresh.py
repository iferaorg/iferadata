import torch

from ifera.file_refresh import aggregate_from_parent_tensor
from ifera.file_utils import make_path, write_tensor_to_gzip, read_tensor_from_gzip
from ifera.enums import Source
from ifera.settings import settings


def test_aggregate_from_parent_tensor(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path))

    parent_steps = 4
    data = torch.zeros((1, parent_steps, 9), dtype=torch.float32)
    data[0, :, 0] = torch.arange(parent_steps)
    data[0, :, 1] = torch.arange(parent_steps) + 0.5
    data[0, :, 2] = torch.arange(parent_steps) - 0.5
    data[0, :, 3] = torch.arange(parent_steps) + 1.0
    data[0, :, 4] = torch.arange(parent_steps)
    data[0, :, 5] = torch.arange(parent_steps) + 0.5
    data[0, :, 6] = torch.arange(parent_steps) - 0.5
    data[0, :, 7] = torch.arange(parent_steps) + 1.0
    data[0, :, 8] = torch.arange(parent_steps) + 1

    parent_path = make_path(Source.TENSOR, "futures", "30m", "CL")
    write_tensor_to_gzip(str(parent_path), data)

    aggregate_from_parent_tensor(
        source="tensor",
        type="futures",
        interval="60m",
        symbol="CL",
        parent_interval="30m",
    )

    result_path = make_path(Source.TENSOR, "futures", "60m", "CL")
    result = read_tensor_from_gzip(str(result_path))

    expected = torch.tensor(
        [
            [
                [0.0, 0.5, -0.5, 1.0, 0.0, 1.5, -0.5, 2.0, 3.0],
                [2.0, 2.5, 1.5, 3.0, 2.0, 3.5, 1.5, 4.0, 7.0],
            ]
        ]
    )
    torch.testing.assert_close(result, expected)
