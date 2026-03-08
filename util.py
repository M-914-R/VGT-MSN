import numpy as np
import os
import torch

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind]
                y_i = self.ys[start_ind:end_ind]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

# =========================

class DataLoader4(DataLoader):
    def __init__(self, xs, ys, dls, dhs, batch_size, pad_with_last_sample=True):

        super().__init__(xs, ys, batch_size, pad_with_last_sample=False)
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding > 0:
                x_padding  = np.repeat(xs[-1:],  num_padding, axis=0)
                y_padding  = np.repeat(ys[-1:],  num_padding, axis=0)
                dl_padding = np.repeat(dls[-1:], num_padding, axis=0)
                dh_padding = np.repeat(dhs[-1:], num_padding, axis=0)
                xs  = np.concatenate([xs,  x_padding],  axis=0)
                ys  = np.concatenate([ys,  y_padding],  axis=0)
                dls = np.concatenate([dls, dl_padding], axis=0)
                dhs = np.concatenate([dhs, dh_padding], axis=0)

        self.xs = xs
        self.ys = ys
        self.do_lows = dls
        self.do_highs = dhs
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                s = self.batch_size * self.current_ind
                e = min(self.size, self.batch_size * (self.current_ind + 1))
                yield (self.xs[s:e], self.ys[s:e], self.do_lows[s:e], self.do_highs[s:e])
                self.current_ind += 1
        return _wrapper()

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std  = float(std) if float(std) != 0.0 else 1.0  # 避免除0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):

    data = {}
    has_do = True
    for category in ["train", "val", "test"]:
        cat_path = os.path.join(dataset_dir, category + ".npz")
        cat_data = np.load(cat_path)
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]
        # 可选键：do_low / do_high
        if ("do_low" in cat_data) and ("do_high" in cat_data):
            data["do_low_" + category]  = cat_data["do_low"]
            data["do_high_" + category] = cat_data["do_high"]
        else:
            has_do = False


    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(),
        std =data["x_train"][..., 0].std()
    )
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

        if has_do:
            data["do_low_"  + category] = scaler.transform(data["do_low_"  + category])
            data["do_high_" + category] = scaler.transform(data["do_high_" + category])

    print("Perform shuffle on the dataset")
    for category in ["train", "val"]:
        perm = torch.randperm(data["x_" + category].shape[0]).numpy()
        data["x_" + category] = data["x_" + category][perm]
        data["y_" + category] = data["y_" + category][perm]
        if has_do:
            data["do_low_"  + category] = data["do_low_"  + category][perm]
            data["do_high_" + category] = data["do_high_" + category][perm]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"]   = DataLoader(data["x_val"],   data["y_val"],   valid_batch_size or batch_size)
    data["test_loader"]  = DataLoader(data["x_test"],  data["y_test"],  test_batch_size  or batch_size)

    if has_do:
        data["train_loader4"] = DataLoader4(
            data["x_train"], data["y_train"], data["do_low_train"], data["do_high_train"], batch_size
        )
        data["val_loader4"] = DataLoader4(
            data["x_val"],   data["y_val"],   data["do_low_val"],   data["do_high_val"],   valid_batch_size or batch_size
        )
        data["test_loader4"] = DataLoader4(
            data["x_test"],  data["y_test"],  data["do_low_test"],  data["do_high_test"],  test_batch_size  or batch_size
        )

    data["scaler"] = scaler
    return data

def MAE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))

def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / (true + 1e-5)))

def RMSE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def WMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))

def R2_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)

    pred = pred.view(-1)
    true = true.view(-1)

    if pred.numel() == 0 or true.numel() == 0:
        return torch.tensor(0.0).to(pred.device)

    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - torch.mean(true)) ** 2)

    if ss_tot < 1e-8:
        return torch.tensor(0.0).to(pred.device)

    return 1 - ss_res / ss_tot

def metric(pred, real):
    mae = MAE_torch(pred, real, 0.0).item()
    mape = MAPE_torch(pred, real, 0.0).item()
    rmse = RMSE_torch(pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()
    r2 = R2_torch(pred, real, 0.0).item()
    return mae, mape, rmse, wmape, r2
