import numpy as np
import pandas as pd
import os


try:
    from vmdpy import VMD
except ImportError as e:
    raise ImportError("请先安装 vmdpy: pip install vmdpy") from e

# =========================
INPUT_LEN   = 12
OUTPUT_LEN  = 1
STRIDE      = 1
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

DATA_DIR       = "./data/cleaned"
SAVE_BASE_DIR  = ("./data/vmd_data")
DATASET_RANGE  = range(1, 5)

DO_COL_NAME = "DO_mgl"

# =========================
# VMD 分解参数
# =========================
VMD_K      = 8       # 拆成 8 个模态
VMD_ALPHA  = 2000.0  # 模态带宽惩罚，越大越窄，常用 2000 左右
VMD_TAU    = 0.0     # 噪声容忍，通常 0
VMD_DC     = 0       # 是否强制 DC 模态（0=否，1=是）; 如趋势很强可改为 1
VMD_INIT   = 1       # 1: 频率均匀初始化；0: 全 0
VMD_TOL    = 1e-7    # 收敛阈值
HIGH_K     = 2       # 高频模态数（排序后最后2个），剩余 6 个为低频

# =========================
def construct_samples(data_raw, x_len=INPUT_LEN, y_len=OUTPUT_LEN, stride=STRIDE):
    """
    data_raw: (T, N, 1)
    return:
      x: (B, x_len, N, 1)
      y: (B, y_len, N, 1)
    """
    T, N, F = data_raw.shape
    xs, ys = [], []
    for t in range(0, T - x_len - y_len + 1, stride):
        xs.append(data_raw[t:t + x_len])
        ys.append(data_raw[t + x_len:t + x_len + y_len])
    return np.stack(xs), np.stack(ys)

def construct_do_windows_like_x(x_windows, do_low, do_high, N, stride=STRIDE):
    """
    x_windows: (B, L, N, 1) —— 已经构造好的 x
    do_low/high: (T_seg,)
    返回: (B, 1, N, L) 与 x 完全对齐
    """
    B, L, _, _ = x_windows.shape
    Ls = np.empty((B, 1, N, L), dtype=np.float32)
    Hs = np.empty_like(Ls)
    for i in range(B):
        start = i * stride
        l_win = do_low[start:start + L]
        h_win = do_high[start:start + L]
        # 边界容错：如果末端不足 L，就用最后一个值补齐
        if l_win.shape[0] < L:
            pad = L - l_win.shape[0]
            if l_win.shape[0] == 0:  # 极端情况保护
                l_win = np.zeros(L, dtype=np.float32)
                h_win = np.zeros(L, dtype=np.float32)
            else:
                l_win = np.concatenate([l_win, np.repeat(l_win[-1:], pad)])
                h_win = np.concatenate([h_win, np.repeat(h_win[-1:], pad)])
        Ls[i, 0, :, :] = np.tile(l_win[None, :], (N, 1))
        Hs[i, 0, :, :] = np.tile(h_win[None, :], (N, 1))
    return Ls, Hs

def save_as_npz(save_path, x, y, do_low, do_high, x_len=INPUT_LEN, y_len=OUTPUT_LEN):
    x_offsets = np.arange(-x_len + 1, 1).reshape(-1, 1)
    y_offsets = np.arange(1, y_len + 1).reshape(-1, 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        x=x, y=y, do_low=do_low, do_high=do_high,
        x_offsets=x_offsets, y_offsets=y_offsets
    )
    print(f" Saved: {save_path}, x:{x.shape}, y:{y.shape}, do_low:{do_low.shape}, do_high:{do_high.shape}")

def split_time_series(arr, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):

    T = arr.shape[0]
    t_tr = int(T * train_ratio)
    t_va = int(T * (train_ratio + val_ratio))
    return arr[:t_tr], arr[t_tr:t_va], arr[t_va:]

# =========================

def vmd_low_high(x_1d, K=VMD_K, high_k=HIGH_K,
                 alpha=VMD_ALPHA, tau=VMD_TAU, DC=VMD_DC, init=VMD_INIT, tol=VMD_TOL):

    import numpy as np
    import pandas as pd
    from vmdpy import VMD

    x = np.asarray(x_1d, dtype=np.float64)
    if np.isnan(x).any():
        x = pd.Series(x).interpolate("linear").bfill().ffill().values

    mu = float(np.mean(x))
    x0 = x - mu

    u, u_hat, omega = VMD(x0, alpha, tau, K, DC, init, tol)   # u: (K,T)

    u = np.asarray(u)
    omega = np.asarray(omega)

    if omega.ndim == 1 and omega.shape[0] == K:
        omega_c = omega
    elif omega.ndim == 2 and omega.shape[0] == K:

        omega_c = omega[:, -1]
    elif omega.size == K:
        omega_c = omega.reshape(K)
    else:
        def center_freq(sig):
            X = np.abs(np.fft.rfft(sig))
            f = np.fft.rfftfreq(sig.size, d=1.0)
            s = X.sum()
            return 0.0 if s == 0 else float((f * X).sum() / s)
        omega_c = np.array([center_freq(u[k]) for k in range(K)], dtype=np.float64)

    order = np.argsort(omega_c)
    k_low = max(K - high_k, 0)

    low  = u[order[:k_low], :].sum(axis=0) if k_low > 0 else np.zeros_like(x0)
    high = u[order[k_low:], :].sum(axis=0) if high_k > 0 else np.zeros_like(x0)

    low  = (low  + mu).astype(np.float32)
    high = (high     ).astype(np.float32)
    return low, high

# =========================
def process_one_dataset(idx: int):
    print(f"\n====  Processing Dataset {idx} ====")
    src_file = os.path.join(DATA_DIR, f"cleaned_dataset{idx}.xlsx")
    assert os.path.exists(src_file), f"Not found: {src_file}"

    df = pd.read_excel(src_file)

    for time_col in ["DateTimeStamp", "datetime", "time", "日期", "时间"]:
        if time_col in df.columns:
            df = df.drop(columns=[time_col])

    cols = df.columns.tolist()
    assert DO_COL_NAME in cols, f"未找到 {DO_COL_NAME} 列，当前列: {cols}"
    N = len(cols)

    raw = df.values.astype(np.float32)  # (T, N)

    raw_tr, raw_va, raw_te = split_time_series(raw)  # (T_seg, N)

    def segment_to_npz(raw_seg: np.ndarray):
        do = raw_seg[:, cols.index(DO_COL_NAME)].astype(np.float32)  # (T_seg,)

        try:
            do_low, do_high = vmd_low_high(do)
        except Exception as e:

            print(f"[WARN] VMD 失败：{e}，改用简单回退分解")
            k = 97
            ma = pd.Series(do).rolling(window=k, center=True, min_periods=1).mean().values.astype(np.float32)
            do_low, do_high = ma, (do - ma).astype(np.float32)

        seg_3d = np.expand_dims(raw_seg, axis=-1)  # (T_seg, N, 1)
        x, y = construct_samples(seg_3d, x_len=INPUT_LEN, y_len=OUTPUT_LEN, stride=STRIDE)

        dl_win, dh_win = construct_do_windows_like_x(x_windows=x, do_low=do_low, do_high=do_high, N=N, stride=STRIDE)

        assert dl_win.shape[0] == x.shape[0] == y.shape[0] == dh_win.shape[0], \
            f"window mismatch: x={x.shape[0]}, y={y.shape[0]}, do_low={dl_win.shape[0]}, do_high={dh_win.shape[0]}"
        return x.astype(np.float32), y.astype(np.float32), dl_win, dh_win

    x_tr, y_tr, dl_tr, dh_tr = segment_to_npz(raw_tr)
    x_va, y_va, dl_va, dh_va = segment_to_npz(raw_va)
    x_te, y_te, dl_te, dh_te = segment_to_npz(raw_te)

    save_dir = os.path.join(SAVE_BASE_DIR, f"dataset{idx}")
    save_as_npz(os.path.join(save_dir, "train.npz"), x_tr, y_tr, dl_tr, dh_tr)
    save_as_npz(os.path.join(save_dir, "val.npz"),   x_va, y_va, dl_va, dh_va)
    save_as_npz(os.path.join(save_dir, "test.npz"),  x_te, y_te, dl_te, dh_te)

if __name__ == "__main__":
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)
    for i in DATASET_RANGE:
        process_one_dataset(i)
