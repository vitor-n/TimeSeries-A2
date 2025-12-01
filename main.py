"""
TiDE vs PatchTST vs AR(1) comparison using NeuralForecast and statsmodels.

- Loads weekly data from `data_updated.csv` (same schema as the notebook).
- Runs a small holdout grid search for TiDE and PatchTST; AR(1) is a fixed baseline.
- Executes walk-forward evaluation (h=4) and prints MAE/MASE by horizon.
- Optionally saves a horizon-1 plot if --plot-path is provided.

Dependencies (install in your env before running):
  pip install "numpy<2.0" neuralforecast pytorch-lightning statsmodels
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST, TiDE
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

# Constants shared with the notebook
HORIZON = 4
INITIAL_TRAIN = 104
SEASONALITY = 52
TEST_SIZE = 30
RANDOM_STATE = 42
FREQUENCY = "W-MON"
FUTR_EXOG = ["month", "weekofyear", "inv", "users"]


def load_data(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)

    df_nf = df.rename(columns={"week": "ds", "volume": "y"})
    df_nf["unique_id"] = "series"
    df_nf["month"] = df_nf["ds"].dt.month.astype(int)
    df_nf["weekofyear"] = df_nf["ds"].dt.isocalendar().week.astype(int)

    y_values = df_nf["y"].values.astype(np.float32)
    train_end_idx = len(df_nf) - TEST_SIZE
    y_holdout = y_values[train_end_idx:]
    futr_base = df_nf[["unique_id", "ds"] + FUTR_EXOG]

    return {
        "df": df,
        "df_nf": df_nf,
        "y_values": y_values,
        "y_holdout": y_holdout,
        "train_end_idx": train_end_idx,
        "futr_base": futr_base,
    }


def calculate_mase(y_true, y_pred, y_train, m=SEASONALITY):
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    mae_naive = np.mean(naive_errors)
    if mae_naive < 1e-6 or np.isnan(mae_naive):
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / mae_naive


def build_tide(params):
    return TiDE(
        h=HORIZON,
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
        num_encoder_layers=params["num_layers"],
        num_decoder_layers=params["num_layers"],
        futr_exog_list=FUTR_EXOG,
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        max_steps=params["max_steps"],
        val_check_steps=params.get("val_check_steps", 100),
        random_seed=RANDOM_STATE,
        alias="TiDE",
    )


def build_patch(params):
    # PatchTST (NeuralForecast) não suporta exógenas futuras nesta versão; usamos univariado
    return PatchTST(
        h=HORIZON,
        input_size=params["input_size"],
        encoder_layers=params["encoder_layers"],
        n_heads=params["n_heads"],
        hidden_size=params["hidden_size"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
        fc_dropout=params["dropout"],
        learning_rate=params["lr"],
        batch_size=params["batch_size"],
        max_steps=params["max_steps"],
        val_check_steps=params.get("val_check_steps", 100),
        random_seed=RANDOM_STATE,
        alias="PatchTST",
    )


def grid_search_nf(builder, grid, df_nf, futr_base, train_end_idx, y_holdout, use_futr):
    train_df = df_nf.iloc[:train_end_idx]
    futr_holdout = (
        futr_base.iloc[train_end_idx : train_end_idx + len(y_holdout)] if use_futr else None
    )

    best_params, best_mae = None, np.inf
    for params in grid:
        model = builder(params)
        nf = NeuralForecast(models=[model], freq=FREQUENCY)
        nf.fit(df=train_df, val_size=0, verbose=False)
        if use_futr:
            preds_df = nf.predict(futr_df=futr_holdout, h=len(y_holdout), verbose=False)
        else:
            preds_df = nf.predict(h=len(y_holdout), verbose=False)
        y_pred = preds_df[model.alias].values
        mae = np.mean(np.abs(y_holdout - y_pred))
        print(f"{model.alias} params {params} | MAE holdout: {mae:.4f}")
        if mae < best_mae:
            best_params, best_mae = params, mae
    return best_params, best_mae


def grid_search_ar1(y_values, train_end_idx, y_holdout):
    y_train = y_values[:train_end_idx]
    model = ARIMA(y_train, order=(1, 0, 0)).fit()
    preds = model.forecast(steps=len(y_holdout))
    mae = np.mean(np.abs(y_holdout - preds))
    print(f"AR(1) MAE holdout: {mae:.4f}")
    return {}, mae


def run_walk_forward_nf(builder, params, alias, df_nf, futr_base, y_values, use_futr):
    rows = []
    for cutoff in range(INITIAL_TRAIN, len(df_nf) - HORIZON + 1):
        train_df = df_nf.iloc[:cutoff]
        futr_df = futr_base.iloc[cutoff : cutoff + HORIZON] if use_futr else None
        target = y_values[cutoff : cutoff + HORIZON]
        train_values = y_values[:cutoff]

        model = builder(params)
        nf = NeuralForecast(models=[model], freq=FREQUENCY)
        nf.fit(df=train_df, val_size=0, verbose=False)
        if use_futr:
            preds_df = nf.predict(futr_df=futr_df, h=HORIZON, verbose=False)
        else:
            preds_df = nf.predict(h=HORIZON, verbose=False)
        preds = preds_df[model.alias].values
        cutoff_time = train_df["ds"].iloc[-1]

        for h in range(HORIZON):
            mase = calculate_mase(target[h], preds[h], train_values, m=SEASONALITY)
            rows.append(
                {
                    "model": alias,
                    "cutoff": cutoff_time,
                    "horizon": h + 1,
                    "y_true": target[h],
                    "y_pred": preds[h],
                    "abs_error": abs(target[h] - preds[h]),
                    "mase": mase,
                }
            )
    return pd.DataFrame(rows)


def run_walk_forward_ar1(df_nf, y_values):
    rows = []
    for cutoff in range(INITIAL_TRAIN, len(df_nf) - HORIZON + 1):
        train = y_values[:cutoff]
        target = y_values[cutoff : cutoff + HORIZON]
        cutoff_time = df_nf["ds"].iloc[cutoff - 1]
        try:
            model = ARIMA(train, order=(1, 0, 0)).fit()
            preds = model.forecast(steps=HORIZON)
        except Exception:
            preds = np.repeat(train[-1], HORIZON)
        for h in range(HORIZON):
            mase = calculate_mase(target[h], preds[h], train, m=SEASONALITY)
            rows.append(
                {
                    "model": "AR(1)",
                    "cutoff": cutoff_time,
                    "horizon": h + 1,
                    "y_true": target[h],
                    "y_pred": preds[h],
                    "abs_error": abs(target[h] - preds[h]),
                    "mase": mase,
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(df_metrics):
    return (
        df_metrics.groupby(["model", "horizon"])
        .agg(MAE=("abs_error", "mean"), MASE=("mase", "mean"))
        .reset_index()
    )


def plot_h1(df, df_tide, df_patch, df_ar1, plot_path: Path | None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["week"], df["volume"], label="Real", color="black", alpha=0.4)

    h1_tide = df_tide[df_tide["horizon"] == 1]
    ax.plot(h1_tide["cutoff"], h1_tide["y_pred"], label="TiDE h=1", color="blue")

    h1_patch = df_patch[df_patch["horizon"] == 1]
    ax.plot(h1_patch["cutoff"], h1_patch["y_pred"], label="PatchTST h=1", color="green")

    h1_ar1 = df_ar1[df_ar1["horizon"] == 1]
    ax.plot(
        h1_ar1["cutoff"],
        h1_ar1["y_pred"],
        label="AR(1) h=1",
        color="red",
        alpha=0.7,
    )

    ax.set_title("Walk-forward horizonte 1")
    ax.legend()
    fig.tight_layout()

    if plot_path:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200)
        print(f"Plot salvo em {plot_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparação TiDE vs PatchTST vs AR(1) sem Darts (NeuralForecast + statsmodels)."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data_updated.csv"),
        help="Caminho para o CSV com colunas week, volume, inv, users.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Se informado, salva o plot de horizonte 1 neste caminho.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.data_path)

    df = data["df"]
    df_nf = data["df_nf"]
    futr_base = data["futr_base"]
    y_values = data["y_values"]
    y_holdout = data["y_holdout"]
    train_end_idx = data["train_end_idx"]

    tide_param_grid = [
        {
            "input_size": 26,
            "hidden_size": 256,
            "dropout": 0.15,
            "num_layers": 2,
            "lr": 1e-3,
            "batch_size": 32,
            "max_steps": 400,
            "val_check_steps": 50,
        }
    ]

    patch_param_grid = [
        {
            "input_size": 36,
            "patch_len": 6,
            "stride": 3,
            "encoder_layers": 3,
            "n_heads": 8,
            "hidden_size": 128,
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 32,
            "max_steps": 400,
            "val_check_steps": 50,
        }
    ]

    print(f"Grid TiDE: {len(tide_param_grid)} combinação(s)")
    best_tide_params, tide_holdout = grid_search_nf(
        build_tide, tide_param_grid, df_nf, futr_base, train_end_idx, y_holdout, use_futr=True
    )

    print(f"Grid PatchTST: {len(patch_param_grid)} combinação(s)")
    best_patch_params, patch_holdout = grid_search_nf(
        build_patch, patch_param_grid, df_nf, futr_base, train_end_idx, y_holdout, use_futr=False
    )

    print("Grid AR(1): 1 combinação")
    best_ar1_params, ar1_holdout = grid_search_ar1(y_values, train_end_idx, y_holdout)

    print(f"Melhor TiDE: {best_tide_params} | MAE holdout={tide_holdout:.4f}")
    print(f"Melhor PatchTST: {best_patch_params} | MAE holdout={patch_holdout:.4f}")
    print(f"Baseline AR(1) MAE holdout={ar1_holdout:.4f}")

    print("Rodando walk-forward TiDE...")
    df_tide = run_walk_forward_nf(
        build_tide, best_tide_params, "TiDE", df_nf, futr_base, y_values, use_futr=True
    )

    print("Rodando walk-forward PatchTST...")
    df_patch = run_walk_forward_nf(
        build_patch, best_patch_params, "PatchTST", df_nf, futr_base, y_values, use_futr=False
    )

    print("Rodando walk-forward AR(1)...")
    df_ar1 = run_walk_forward_ar1(df_nf, y_values)

    df_all = pd.concat([df_tide, df_patch, df_ar1], ignore_index=True)
    metrics = summarize_metrics(df_all)
    print("\nMAE/MASE por horizonte:")
    print(metrics)

    plot_h1(df, df_tide, df_patch, df_ar1, args.plot_path)


if __name__ == "__main__":
    main()
