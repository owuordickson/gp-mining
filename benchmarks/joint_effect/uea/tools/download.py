"""Read the official UEA train and test splits from local files or the archive."""

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

DATASETS = (
    "BasicMotions",
    "RacketSports",
    "NATOPS",
    "AtrialFibrillation",
    "Epilepsy",
    "ERing",
    "FingerMovements",
    "ArticularyWordRecognition",
    "Libras",
    "LSST",
    "PenDigits",
    "AppliancesEnergy_disc",
    "EyesOpenShut",
    "Handwriting",
    "IRDS-EFL",
    "JapaneseVowels_eq",
    "SPHERE-WUS",
    "UIPRMD-SSE-C"
)
BASE_URL = "https://www.timeseriesclassification.com/aeon-toolkit"
_MAX_BYTES = 64 * 1024 * 1024


def _parse_ts(content: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one labelled multivariate .ts file into sequences and labels.

    Decode UTF-8 text, accept an optional byte-order mark, and read
    the header up to @data. Require class labels and untimestamped
    observations. Parse each channel's comma-separated values,
    transpose each case into time/channel order, and stack cases
    after checking equal shapes and finite values.

    Args:
        content: Raw bytes of one training or test split.
        name: Split filename included in parsing error messages.

    Returns:
        A float64 array shaped (cases, observations, channels) and
        a one-dimensional array of the original string labels.

    Raises:
        ValueError: Headers or records are malformed, values are
            non-finite, shapes differ, or the file contains no cases.
    """
    headers: dict[str, str] = {}
    cases = []
    labels = []
    reading_data = False
    for line_number, raw in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not reading_data:
            key, _, value = line.partition(" ")
            key = key.casefold()
            if key == "@data":
                if headers.get("@timestamps", "").casefold() != "false":
                    raise ValueError(f"{name}: expected an untimestamped .ts file")
                if headers.get("@classlabel", "").casefold().split()[:1] != ["true"]:
                    raise ValueError(f"{name}: expected class labels")
                reading_data = True
            elif key.startswith("@"):
                headers[key] = value.strip()
            else:
                raise ValueError(f"{name}:{line_number}: malformed .ts header")
            continue

        values, separator, label = line.rpartition(":")
        if not separator or not label.strip():
            raise ValueError(f"{name}:{line_number}: missing values or label")
        try:
            case = np.array(
                [
                    [float(point) for point in channel.split(",")]
                    for channel in values.split(":")
                ],
                dtype=np.float64,
            ).T
        except ValueError as exc:
            raise ValueError(f"{name}:{line_number}: invalid case values") from exc
        if not np.isfinite(case).all() or (cases and case.shape != cases[0].shape):
            raise ValueError(
                f"{name}:{line_number}: expected equal-length finite cases"
            )
        cases.append(case)
        labels.append(label.strip())

    if not cases:
        raise ValueError(f"{name}: no labelled cases")
    return np.stack(cases), np.asarray(labels)


def _zip_splits(source: Path | BytesIO, names: tuple[str, str]) -> tuple[bytes, bytes]:
    """Read the two requested split files from a ZIP archive.

    Match each requested basename case-insensitively, allowing nested
    directories and either slash style. Require exactly one match
    per name and check its uncompressed size against _MAX_BYTES
    before reading it.

    Args:
        source: ZIP path or in-memory BytesIO archive.
        names: Requested split basenames, in return order.

    Returns:
        The two split payloads as bytes.

    Raises:
        ValueError: A requested member is missing, duplicated, or
            exceeds the configured size limit.
    """
    with ZipFile(source) as archive:
        parts = []
        for name in names:
            matches = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and info.filename.replace("\\", "/").rsplit("/", 1)[-1].casefold()
                == name.casefold()
            ]
            if len(matches) != 1 or matches[0].file_size > _MAX_BYTES:
                raise ValueError(f"Expected one bounded ZIP member named {name}")
            parts.append(archive.read(matches[0]))
    return parts[0], parts[1]


def download_n_split(
    dataset: str, data_dir: Path, *, download: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset's published training and test recordings.

    Look for extracted .ts files in the dataset subfolder and data_dir,
    then try the cached ZIP. When download is enabled and local data
    is absent, fetch a bounded archive from BASE_URL. Parse both
    splits and check their channel/time dimensions and class sets.
    Cache a successful download by replacing a temporary file after
    validation.

    Args:
        dataset: Dataset name listed in DATASETS.
        data_dir: Folder containing extracted splits or cached ZIPs.
        download: Whether missing data may be fetched from the archive.

    Returns:
        x_train, y_train, x_test, y_test. Sequence arrays use
        (cases, observations, channels) order, and labels retain
        their original string values and published split membership.

    Raises:
        ValueError: The name is unsupported, an archive or split is
            invalid, or the split dimensions or class sets differ.
        FileNotFoundError: Local data is absent and download is disabled.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown UEA dataset {dataset!r}")
    names = (f"{dataset}_TRAIN.ts", f"{dataset}_TEST.ts")
    cache = data_dir / f"{dataset}.zip"
    payload = None
    for directory in (data_dir / dataset, data_dir):
        train, test = (directory / name for name in names)
        if train.is_file() and test.is_file():
            raw_train, raw_test = train.read_bytes(), test.read_bytes()
            break
    else:
        if cache.is_file():
            raw_train, raw_test = _zip_splits(cache, names)
        elif download:
            with urlopen(f"{BASE_URL}/{dataset}.zip", timeout=30) as response:
                payload = response.read(_MAX_BYTES + 1)
            if len(payload) > _MAX_BYTES:
                raise ValueError(f"{dataset}: archive exceeds the size limit")
            raw_train, raw_test = _zip_splits(BytesIO(payload), names)
        else:
            raise FileNotFoundError(
                f"Place both {dataset} .ts splits or {cache.name} in "
                f"{data_dir}, or pass --download"
            )

    x_train, y_train = _parse_ts(raw_train, names[0])
    x_test, y_test = _parse_ts(raw_test, names[1])
    if x_train.shape[1:] != x_test.shape[1:]:
        raise ValueError(f"{dataset}: incompatible train and test shapes")
    if set(y_train) != set(y_test):
        raise ValueError(f"{dataset}: train and test have different class sets")

    if payload is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=data_dir, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(payload)
                handle.close()
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
    return x_train, y_train, x_test, y_test
