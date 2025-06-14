import datetime
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from ifera import s3_utils, ConfigManager, BaseInstrumentConfig, InstrumentConfig


@pytest.fixture
def sample_vector():
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_vector_masked_all_true(sample_vector):
    mask = torch.ones_like(sample_vector, dtype=torch.bool)
    return sample_vector, mask


@pytest.fixture
def sample_vector_masked_partial():
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, False, True, False, True])
    return data, mask


@pytest.fixture
def ohlcv_two_points():
    return torch.tensor(
        [[10.0, 12.0, 8.0, 10.0, 100.0], [11.0, 13.0, 9.0, 12.0, 150.0]]
    )


@pytest.fixture
def ohlcv_two_points_masked(ohlcv_two_points):
    data = ohlcv_two_points.unsqueeze(0)
    mask = torch.ones(data.shape[:-1], dtype=torch.bool)
    return data, mask


@pytest.fixture
def ohlcv_single_date():
    return torch.tensor(
        [
            [
                [10.0, 12.0, 8.0, 10.0, 100.0],
                [11.0, 13.0, 9.0, 12.0, 150.0],
                [12.0, 14.0, 10.0, 13.0, 120.0],
            ]
        ]
    )


@pytest.fixture
def ohlcv_single_date_masked(ohlcv_single_date):
    mask = torch.ones(ohlcv_single_date.shape[:-1], dtype=torch.bool)
    return ohlcv_single_date, mask


@pytest.fixture
def dummy_progress(monkeypatch):
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def update(self, _):
            pass

        def close(self):
            pass

    def _factory(*args, **kwargs):
        return DummyProgress(*args, **kwargs)

    monkeypatch.setattr(s3_utils, "tqdm", _factory)
    return DummyProgress


@pytest.fixture
def mock_s3(monkeypatch):
    client = MagicMock()
    wrapper = SimpleNamespace(
        client=client,
        cache=True,
        last_modified={},
        cached_prefixes=set(),
    )

    def _populate_cache(prefix: str) -> None:
        response = client.list_objects_v2(
            Bucket=s3_utils.settings.S3_BUCKET, Prefix=prefix
        )
        for obj in response.get("Contents", []):
            wrapper.last_modified[obj["Key"]] = obj.get(
                "LastModified", datetime.datetime.now(tz=datetime.timezone.utc)
            )
        wrapper.cached_prefixes.add(prefix)

    wrapper._populate_cache = _populate_cache
    monkeypatch.setattr(s3_utils, "S3ClientSingleton", lambda cache=True: wrapper)
    return wrapper


@pytest.fixture
def mock_github(monkeypatch):
    """Provide a mocked GitHub client for offline tests."""
    from ifera import github_utils

    client = MagicMock()
    wrapper = SimpleNamespace(github_client=client)
    monkeypatch.setattr(github_utils, "GitHubClientSingleton", lambda: wrapper)
    github_utils.check_github_file_exists.cache_clear() # type: ignore
    github_utils.get_github_last_modified.cache_clear() # type: ignore
    return wrapper


# Instrument Configuration
@pytest.fixture
def config_manager() -> ConfigManager:
    return ConfigManager(
        instruments_filename="data/test_instruments.yml",
        brokers_filename="data/test_brokers.yml",
    )


@pytest.fixture
def base_instrument_config(config_manager: ConfigManager) -> BaseInstrumentConfig:
    return config_manager.get_base_instrument_config(symbol="CL", interval="30m")


@pytest.fixture
def instrument_config(config_manager: ConfigManager) -> InstrumentConfig:
    return config_manager.get_config(broker_name="IBKR", symbol="CL", interval="30m")
