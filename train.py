import pandas as pd
import argparse
import time
import util
from util import *
import random
from model import VGTMSN
from ranger import Ranger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="")
parser.add_argument("--data", type=str, default="dataset1", help="dataset directory path")
parser.add_argument("--input_dim", type=int, default=1, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=8, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=1, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type= float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument("--epochs", type=int, default=200, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument("--save", type=str, default="./log/" + str(time.strftime("%Y-%m-%d-%H_%M_%S")) + "-",
                    help="save path")
parser.add_argument("--es_patience", type=int, default=50,
                    help="quit if no improvement after this many iterations")
args = parser.parse_args()

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

class trainer:
    def __init__(
            self,
            scaler,
            input_dim,
            channels,
            num_nodes,
            input_len,
            output_len,
            dropout,
            lrate,
            wdecay,
            device,
    ):
        self.model = VGTMSN(
            device, input_dim, channels, num_nodes, input_len, output_len, dropout,
        )

        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5
        print(self.model)

    def train(self, input, real_val, dataset_index, do_low=None, do_high=None):
        self.model.train()
        self.optimizer.zero_grad()
        try:
            output = self.model(input, dataset_index, do_low=do_low, do_high=do_high)
        except TypeError:
            output = self.model(input, dataset_index)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        predict = predict[:, :, 0, :]
        real = real[:, :, 0, :]

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        r2 = util.R2_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape, r2

    def eval(self, input, real_val, dataset_index, do_low=None, do_high=None):
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(input, dataset_index, do_low=do_low, do_high=do_high)
            except TypeError:
                output = self.model(input, dataset_index)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        predict = predict[:, :, 0, :]
        real = real[:, :, 0, :]

        loss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        r2 = util.R2_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape, r2

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)

def main():
    seed_it(6666)

    data = args.data
    dataset_index = {"dataset1":1, "dataset2":2, "dataset3":3, "dataset4":4}.get(data, None)
    if dataset_index is None:
        raise ValueError("Unknown dataset name. Please use one of: dataset1, dataset2, dataset3, dataset4")

    device = torch.device(args.device)

    dataloader = util.load_dataset(
        f"data/vmd_data/{data}", args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]


    use_four = ("train_loader4" in dataloader) and ("val_loader4" in dataloader) and ("test_loader4" in dataloader)


    path = args.save + data + "/"

    early_stopping = EarlyStopping(
        patience=args.es_patience,
        verbose=True,
        save_path=path + "best_model.pth"
    )

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []
    train_epoch_loss = []
    valid_epoch_loss = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        args.channels,
        args.num_nodes,
        args.input_len,
        args.output_len,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device
    )

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        # ================= train =================
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []
        train_r2 = []

        t1 = time.time()
        if use_four:
            it = dataloader["train_loader4"].get_iterator()
            for iter, (x, y, do_l, do_h) in enumerate(it):
                trainx = torch.Tensor(x).to(device).transpose(1, 3)
                trainy = torch.Tensor(y).to(device).transpose(1, 3)
                do_lowx  = torch.Tensor(do_l).to(device)
                do_highx = torch.Tensor(do_h).to(device)
                metrics = engine.train(trainx, trainy[:, 0, :, :], dataset_index, do_low=do_lowx, do_high=do_highx)
                train_loss.append(metrics[0]); train_mape.append(metrics[1])
                train_rmse.append(metrics[2]); train_wmape.append(metrics[3]); train_r2.append(metrics[4])
        else:
            it = dataloader["train_loader"].get_iterator()
            for iter, (x, y) in enumerate(it):
                trainx = torch.Tensor(x).to(device).transpose(1, 3)
                trainy = torch.Tensor(y).to(device).transpose(1, 3)
                metrics = engine.train(trainx, trainy[:, 0, :, :], dataset_index)
                train_loss.append(metrics[0]); train_mape.append(metrics[1])
                train_rmse.append(metrics[2]); train_wmape.append(metrics[3]); train_r2.append(metrics[4])

        if iter % args.print_every == 0:
            log = "Iter: {:03d}, Train Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, WMAPE: {:.4f}, R2: {:.4f}"
            print(log.format(iter, train_loss[-1], train_rmse[-1], train_mape[-1], train_wmape[-1], train_r2[-1]), flush=True)

        t2 = time.time()
        print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # ================= validation =================
        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []
        valid_r2 = []

        s1 = time.time()
        if use_four:
            it = dataloader["val_loader4"].get_iterator()
            for iter, (x, y, do_l, do_h) in enumerate(it):
                testx = torch.Tensor(x).to(device).transpose(1, 3)
                testy = torch.Tensor(y).to(device).transpose(1, 3)
                do_lowx  = torch.Tensor(do_l).to(device)
                do_highx = torch.Tensor(do_h).to(device)
                metrics = engine.eval(testx, testy[:, 0, :, :], dataset_index, do_low=do_lowx, do_high=do_highx)
                valid_loss.append(metrics[0]); valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2]); valid_wmape.append(metrics[3]); valid_r2.append(metrics[4])
        else:
            it = dataloader["val_loader"].get_iterator()
            for iter, (x, y) in enumerate(it):
                testx = torch.Tensor(x).to(device).transpose(1, 3)
                testy = torch.Tensor(y).to(device).transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :], dataset_index)
                valid_loss.append(metrics[0]); valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2]); valid_wmape.append(metrics[3]); valid_r2.append(metrics[4])
        s2 = time.time()
        print("Epoch: {:03d}, Inference Time: {:.4f} secs".format(i, (s2 - s1)))
        val_time.append(s2 - s1)


        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_r2 = np.mean(train_r2)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_r2 = np.mean(valid_r2)

        train_epoch_loss.append(mtrain_loss)
        valid_epoch_loss.append(mvalid_loss)
        his_loss.append(mvalid_loss)

        train_m = dict(
            train_loss=mtrain_loss,
            train_rmse=mtrain_rmse,
            train_mape=mtrain_mape,
            train_wmape=mtrain_wmape,
            train_r2=mtrain_r2,
            valid_loss=mvalid_loss,
            valid_rmse=mvalid_rmse,
            valid_mape=mvalid_mape,
            valid_wmape=mvalid_wmape,
            valid_r2=mvalid_r2,
        )
        result.append(pd.Series(train_m))

        print("Epoch: {:03d}, Train Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, WMAPE: {:.4f}, R2: {:.4f}".format(
            i, mtrain_loss, mtrain_rmse, mtrain_mape, mtrain_wmape, mtrain_r2
        ), flush=True)
        print("Epoch: {:03d}, Valid Loss: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, WMAPE: {:.4f}, R2: {:.4f}".format(
            i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape, mvalid_r2
        ), flush=True)
        if hasattr(engine.model, "last_alpha_mean") and engine.model.last_alpha_mean is not None:
            print(f"alpha(mean) = {engine.model.last_alpha_mean:.3f}")

        early_stopping(mvalid_loss, engine.model)
        if early_stopping.early_stop and i >= 50:
            print(f"Early stopping at epoch {i}")
            break

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")


    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(train_epoch_loss, label="Train Loss")
    plt.plot(valid_epoch_loss, label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Validation Loss Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("results/loss_curve.png")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # ================= test =================
    print("Training ends")
    print("The best valid loss:", round(np.min(his_loss), 4))
    print("The best epoch:", np.argmin(his_loss) + 1)


    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]


    if use_four:
        it = dataloader["test_loader4"].get_iterator()
        for iter, (x, y, do_l, do_h) in enumerate(it):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            do_lowx = torch.Tensor(do_l).to(device)
            do_highx = torch.Tensor(do_h).to(device)
            with torch.no_grad():
                try:
                    preds = engine.model(testx, dataset_index, do_low=do_lowx, do_high=do_highx).transpose(1, 3)
                except TypeError:
                    preds = engine.model(testx, dataset_index).transpose(1, 3)
            outputs.append(preds.squeeze())
    else:
        it = dataloader["test_loader"].get_iterator()
        for iter, (x, y) in enumerate(it):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx, dataset_index).transpose(1, 3)
            outputs.append(preds.squeeze())


    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]


    amae, amape, armse, awmape, ar2 = [], [], [], [], []
    metrics_rows = []
    true_values = []
    pred_values = []


    for i in range(args.output_len):
        pred = scaler.inverse_transform(yhat[:, 0].reshape(-1, 1)).squeeze()
        real = realy[:, 0, i]
        metrics = util.metric(pred, real)

        mae, mape, rmse, wmape, r2 = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]


        true_values.append(real.detach().cpu().numpy())
        pred_values.append(pred.detach().cpu().numpy())

        log = "Horizon {:02d} | Test MAE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f} | WMAPE: {:.4f} | R²: {:.4f}"
        print(log.format(i + 1, float(mae), float(rmse), float(mape), float(wmape), float(r2)))


        metrics_rows.append({
            "Horizon": int(i + 1),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "WMAPE": float(wmape),
            "R2": float(r2),
        })

        amae.append(float(mae))
        amape.append(float(mape))
        armse.append(float(rmse))
        awmape.append(float(wmape))
        ar2.append(float(r2))


    print("\n[Average over {} horizons] Test MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}, WMAPE: {:.4f}, R²: {:.4f}".format(
        args.output_len, np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape), np.mean(ar2)
    ))


    metrics_rows.append({
        "Horizon": "AVG",
        "MAE": float(np.mean(amae)),
        "RMSE": float(np.mean(armse)),
        "MAPE": float(np.mean(amape)),
        "WMAPE": float(np.mean(awmape)),
        "R2": float(np.mean(ar2)),
    })


    metrics_df = pd.DataFrame(metrics_rows)


    true_values = np.array(true_values).T
    pred_values = np.array(pred_values).T
    df_true_pred = pd.DataFrame(
        np.column_stack([true_values, pred_values]),
        columns=[f"True_{i + 1}" for i in range(args.output_len)] +
                [f"Pred_{i + 1}" for i in range(args.output_len)]
    )


    os.makedirs(path, exist_ok=True)
    with pd.ExcelWriter(f"{path}/test_results.xlsx", engine="openpyxl") as writer:
        df_true_pred.to_excel(writer, sheet_name="True vs Pred", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
