from enum import Enum


class Scheme(Enum):
    """Enumeration of storage schemes or protocols."""

    FILE = "file"
    S3 = "s3"
    GITHUB = "github"


class Source(Enum):
    """Enumeration of data processing states."""

    RAW = "raw"
    PROCESSED = "processed"
    TENSOR = "tensor"


extension_map = {
    Source.RAW: ".zip",
    Source.PROCESSED: ".zip",
    Source.TENSOR: ".pt",
}
