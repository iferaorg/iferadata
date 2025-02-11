# ifera

A Python library for processing financial instrument data.

## Features

- Load and process financial data from various sources
- Support for raw and processed data formats
- AWS S3 integration for data storage
- Configurable data processing pipeline
- PyTorch tensor support

## Installation

```bash
pip install -e .
```

## Usage

```python
from ifera import load_data, InstrumentData, settings

# Configure settings (optionally via environment variables)
settings.DATA_FOLDER = "data"
settings.S3_BUCKET = "your-raw-bucket"
settings.S3_BUCKET_PROCESSED = "your-processed-bucket"

# Load an instrument configuration
instrument = InstrumentData(
    symbol="BTCUSD",
    currency="USD",
    type="crypto",
    broker="binance",
    interval="1m",
    # ... other required fields ...
)

# Load processed data
df = load_data(raw=False, instrument=instrument)

# Or load as PyTorch tensor
tensor = load_data_tensor(instrument=instrument)
```

## Project Structure

```
ifera/
├── ifera/
│   ├── __init__.py      # Package initialization
│   ├── config.py        # Configuration settings
│   ├── models.py        # Data models
│   ├── data_loading.py  # Data loading functionality
│   ├── data_processing.py # Data processing functions
│   └── s3_utils.py      # AWS S3 utilities
└── setup.py            # Package setup configuration
```

## Requirements

- Python >= 3.9
- pandas
- numpy
- PyTorch
- boto3
- pydantic
- pydantic-settings
- tqdm