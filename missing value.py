import pandas as pd
from pathlib import Path

# --------------------

data_folder = Path("./data/raw data")
output_folder = Path("./data/cleaned")
output_folder.mkdir(parents=True, exist_ok=True)

FREQ_MIN = 15
MAX_GAP_MIN = 120
LIMIT_STEPS = MAX_GAP_MIN // FREQ_MIN

# --------------------

def _detect_datetime_col(df: pd.DataFrame):

    candidates = ["datetime", "DateTime", "timestamp", "Timestamp", "time", "Time", df.columns[0]]
    for col in candidates:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().mean() > 0.9:
                return col
    return None

def fill_missing_values_time_phase(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    tcol = _detect_datetime_col(df)
    if tcol is not None:
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
        has_time_index = True
    else:
        has_time_index = False

    num_cols = df.select_dtypes(include="number").columns

    if has_time_index:
        df[num_cols] = df[num_cols].interpolate(
            method="time",
            limit=LIMIT_STEPS if LIMIT_STEPS > 0 else None,
            limit_direction="both"
        )
    else:
        df[num_cols] = df[num_cols].interpolate(
            method="linear",
            limit=LIMIT_STEPS if LIMIT_STEPS > 0 else None,
            limit_direction="both"
        )

    if has_time_index:
        slot = df.index.hour * 60 + df.index.minute
        phase_mean = df.groupby(slot)[num_cols].transform("mean")
        df[num_cols] = df[num_cols].fillna(phase_mean)

    df[num_cols] = df[num_cols].ffill().bfill()

    if has_time_index:
        df = df.reset_index()

    return df

# --------------------

for file_path in sorted(data_folder.glob("*.xlsx")):
    if file_path.name.startswith("~$"):   # 跳过临时文件
        continue

    df = pd.read_excel(file_path)

    print(f"缺失统计 - {file_path.name}:\n{df.isnull().sum()}\n")

    df_cleaned = fill_missing_values_time_phase(df)

    save_path = output_folder / f"cleaned_{file_path.name}"
    df_cleaned.to_excel(save_path, index=False)
    print(f"已保存处理后的文件: {save_path}\n")
