"""Section 5 — Dataset upload / loading."""
import os
import pandas as pd


class DatasetLoadError(Exception):
    pass


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(file_path: str) -> tuple[pd.DataFrame, dict]:
    """Load a CSV/XLSX file into a DataFrame. Never silently fails."""
    if not os.path.exists(file_path):
        raise DatasetLoadError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetLoadError(
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    file_size = os.path.getsize(file_path)

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        raise DatasetLoadError(f"Failed to parse file: {e}") from e

    if df.empty or df.shape[1] == 0:
        raise DatasetLoadError("The uploaded dataset is empty or has no columns.")

    file_info = {
        "file_name": os.path.basename(file_path),
        "file_type": ext.replace(".", "").upper(),
        "file_size_kb": round(file_size / 1024, 2),
        "rows": df.shape[0],
        "columns": df.shape[1],
    }
    return df, file_info