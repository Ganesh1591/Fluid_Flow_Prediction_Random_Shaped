import os
import re
import glob
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# MATPLOTLIB GLOBAL STYLE  (white background + Garuda font)
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "Garuda",
    "font.weight":       "bold",
    "font.size":         13,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "text.color":        "#111111",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

# ─────────────────────────────────────────────
# PAGE CONFIG  (light / white Streamlit theme)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Microchannel Flow Neural Operator",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        background-color: #ffffff;
        color: #111111;
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'DM Mono', monospace;
        color: #111111;
        letter-spacing: -0.5px;
    }
    .stButton > button {
        background: #111111;
        color: #ffffff;
        border: none;
        border-radius: 3px;
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        padding: 8px 22px;
    }
    .stButton > button:hover { background: #333333; }
    section[data-testid="stSidebar"] {
        background: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }
    .metric-box {
        background: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-val {
        font-family: 'DM Mono', monospace;
        font-size: 21px;
        color: #111111;
        font-weight: 600;
    }
    .metric-lbl { font-size: 11px; color: #666666; margin-top: 4px; }
    .summary-box {
        background: #f9f9f9;
        border-left: 4px solid #111111;
        padding: 20px 24px;
        border-radius: 4px;
        font-size: 15px;
        line-height: 1.85;
        color: #222222;
    }
    .tag {
        display: inline-block;
        background: #111111;
        color: #ffffff;
        border-radius: 3px;
        padding: 2px 8px;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        margin-right: 6px;
    }
    hr { border-color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTOENCODER CLASS
# ─────────────────────────────────────────────
class AutoencoderModel_10L(tf.keras.Model):
    def __init__(self, activation="gelu"):
        super().__init__()
        act = activation
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation=act),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(128, activation=act),
            tf.keras.layers.Dense(64,  activation=act),
            tf.keras.layers.Dense(32,  activation=act),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64,  activation=act),
            tf.keras.layers.Dense(128, activation=act),
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(512, activation=act),
            tf.keras.layers.Dense(4),
        ])

    def call(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────
# LOAD MODEL FROM DISK
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_disk(folder, model_name, activation):
    weights_path = os.path.join(folder, f"{model_name}_weights.h5")
    xyt_scaler   = joblib.load(os.path.join(folder, f"{model_name}_xyt_scaler.pkl"))
    out_scaler   = joblib.load(os.path.join(folder, f"{model_name}_out_scaler.pkl"))
    model        = AutoencoderModel_10L(activation=activation)
    model(np.zeros((1, 4), dtype=np.float32))   # build
    model.load_weights(weights_path)
    return model, xyt_scaler, out_scaler


# ─────────────────────────────────────────────
# DATA LOADING  (.dat file parser)
# ─────────────────────────────────────────────
def parse_dat_file(uploaded_file):
    """
    Whitespace-delimited .dat  (no header).
    Expected column order:
      col 0 -> X
      col 1 -> Y
      col 2 -> Z
      col 3 -> RHO  (used as 4th model input)
      col 4 -> U    (ground truth)
      col 5 -> V    (ground truth)
      col 6 -> W    (ground truth)
      col 7 -> Um   (ground truth)
    T is NEVER in the file — always injected from the filename.
    """
    raw   = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        try:
            float(s.split()[0])
            lines.append(s)
        except ValueError:
            continue

    if not lines:
        return None, "No numeric data found in file."

    data  = np.array([list(map(float, ln.split())) for ln in lines])
    ncols = data.shape[1]

    if ncols < 3:
        return None, f"Only {ncols} columns found — need at least 3 (X, Y, Z)."

    df          = pd.DataFrame()
    df["X"]     = data[:, 0]
    df["Y"]     = data[:, 1]
    df["Z"]     = data[:, 2]
    df["RHO"]   = data[:, 3] if ncols > 3 else np.nan
    df["U"]     = data[:, 4] if ncols > 4 else np.nan
    df["V"]     = data[:, 5] if ncols > 5 else np.nan
    df["W"]     = data[:, 7] if ncols > 7 else np.nan
    df["Um"]    = data[:, 7] if ncols > 7 else np.nan
    return df, None


def extract_timestamp(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def build_xyt_array(df, timestamp):
    """
    Assemble the (N, 4) input array: [X, Y, RHO, T]
    """
    n   = len(df)
    X   = df["X"].values.reshape(-1, 1)
    Y   = df["Y"].values.reshape(-1, 1)
    RHO = df["RHO"].values.reshape(-1, 1) if not df["RHO"].isna().all() else np.zeros((n, 1))
    T   = np.full((n, 1), timestamp)
    return np.hstack([X, Y, RHO, T]).astype(np.float32)


def build_out_array(df):
    """
    Assemble the (N, 4) ground-truth array: [U, V, W, Um]
    Returns None columns as NaN where missing.
    """
    def col(name):
        return df[name].values.reshape(-1, 1) if name in df.columns and not df[name].isna().all() \
               else np.full((len(df), 1), np.nan)
    return np.hstack([col("U"), col("V"), col("W"), col("Um")]).astype(np.float32)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_all_models(models_dict, xyt_array):
    """
    Returns { model_name: np.ndarray (N, 4) }  [U, V, W, Um]
    """
    results = {}
    for name, (model, xyt_sc, out_sc) in models_dict.items():
        norm          = xyt_sc.transform(xyt_array)
        y_norm        = model(tf.convert_to_tensor(norm, dtype=tf.float32)).numpy()
        results[name] = out_sc.inverse_transform(y_norm)
    return results


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics_table(out_true, preds_dict):
    """
    Returns a DataFrame with MSE / MAE / RMSE / MaxAE / R2 / RelErr
    for each variable × model combination (where ground truth is not NaN).
    """
    var_names = ["U", "V", "W", "Um"]
    rows = []
    for vi, vname in enumerate(var_names):
        y_true = out_true[:, vi]
        if np.isnan(y_true).all():
            continue
        for mname, pred in preds_dict.items():
            y_pred = pred[:, vi]
            mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if mask.sum() == 0:
                continue
            yt, yp = y_true[mask], y_pred[mask]
            mse    = np.mean((yt - yp) ** 2)
            mae    = np.mean(np.abs(yt - yp))
            rmse   = np.sqrt(mse)
            maxae  = np.max(np.abs(yt - yp))
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            r2     = 1.0 - ss_res / (ss_tot + 1e-12)
            rel_e  = np.mean(np.abs(yt - yp) / (np.abs(yt) + 1e-12)) * 100
            rows.append({
                "Variable": vname, "Model": mname,
                "MSE": mse, "MAE": mae, "RMSE": rmse,
                "Max AE": maxae, "R2 Score": r2, "Rel Error (%)": rel_e,
            })
    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────
# CORE PLOTTING FUNCTION
# (style identical to plot_4timesteps_allmodels_with_inline_errors)
# ─────────────────────────────────────────────
def plot_inline_comparison(
    xyt_raw,
    out_true,
    preds_dict,
    time_levels,
    labels=("U", "V", "W"),
    levels_main=150,
    levels_err=150,
    xlim=(-1.1, 1.1),
    ylim=(-1.1, 1.1),
    row_height=3.2,
    col_width=4.4,
    dpi=150,
    out_fig=None,
    out_csv=None,
    save_files=False,
):
    """
    ONE BIG FIGURE — exact layout from the reference script:

      Rows  : nT * nFields  (e.g. 4 timesteps x 3 fields = 12 rows)
      Cols  : 1 (CFD) + 2 * n_models  (pred + err per model)

    White background throughout. Colorbar on every subplot.
    Font: Garuda bold (falls back gracefully if not installed).
    """
    model_names = list(preds_dict.keys())
    n_models    = len(model_names)
    nT          = len(time_levels)
    nF          = len(labels)
    nrows       = nT * nF
    ncols       = 1 + 2 * n_models

    fig_w = col_width * ncols
    fig_h = row_height * nrows

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    if axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    all_df   = []
    row_base = 0

    # ── helper: add colorbar ──────────────────
    def _cf(ax, Xg, Yg, Z, cmap, title, vmin=None, vmax=None, levels=150):
        try:
            kw = dict(levels=levels, cmap=cmap)
            if vmin is not None:
                kw["vmin"] = vmin
                kw["vmax"] = vmax
            im = ax.contourf(Xg, Yg, Z, **kw)
            cb = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85, format="%.3g")
            cb.ax.tick_params(labelsize=9, colors="#333333")
            cb.outline.set_edgecolor("#cccccc")
        except Exception:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#888888")
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color="#111111", pad=5)
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")

    # ── iterate over timesteps ─────────────────
    for t in time_levels:
        # per-timestep xlim / ylim override
        xlim_t = (-0.1, 2.1) if t == 3 else xlim
        ylim_t = (-0.1, 2.1) if t == 3 else ylim

        mask   = xyt_raw[:, 3] == t
        xyt_t  = xyt_raw[mask]
        out_t  = out_true[mask]

        if xyt_t.shape[0] == 0:
            st.warning(f"No data found for t = {t}. Skipping.")
            row_base += nF
            continue

        # build dataframe for this timestep
        df_t = pd.DataFrame({
            "T": np.full(xyt_t.shape[0], t),
            "X": xyt_t[:, 0],
            "Y": xyt_t[:, 1],
            "U_true": out_t[:, 0],
            "V_true": out_t[:, 1],
            "W_true": out_t[:, 2],
        })

        for name in model_names:
            key = name.replace(" ", "_").replace("-", "_")
            pred = preds_dict[name]
            df_t[f"U_pred_{key}"] = pred[mask, 0]
            df_t[f"V_pred_{key}"] = pred[mask, 1]
            df_t[f"W_pred_{key}"] = pred[mask, 2]

        all_df.append(df_t)

        # build grid
        x_vals = np.sort(df_t["X"].unique())
        y_vals = np.sort(df_t["Y"].unique())
        Xg, Yg = np.meshgrid(x_vals, y_vals)

        def _grid(field):
            try:
                return df_t.pivot(index="Y", columns="X", values=field).values
            except Exception:
                return np.full((len(y_vals), len(x_vals)), np.nan)

        # ── draw rows for this timestep ─────────
        for fi, lab in enumerate(labels):
            r    = row_base + fi
            Zt   = _grid(f"{lab}_true")
            vmin = np.nanmin(Zt)
            vmax = np.nanmax(Zt)

            # Column 0 — CFD ground truth
            axs[r, 0].set_xlim(*xlim_t)
            axs[r, 0].set_ylim(*ylim_t)
            _cf(axs[r, 0], Xg, Yg, Zt, "jet",
                f"CFD  {lab}  (t={t})", vmin=vmin, vmax=vmax,
                levels=levels_main)

            # Columns 1..6 — pred + err per model
            for mi, name in enumerate(model_names):
                key      = name.replace(" ", "_").replace("-", "_")
                Zp       = _grid(f"{lab}_pred_{key}")
                Ze       = np.abs(Zt - Zp)
                col_pred = 1 + 2 * mi
                col_err  = col_pred + 1

                axs[r, col_pred].set_xlim(*xlim_t)
                axs[r, col_pred].set_ylim(*ylim_t)
                _cf(axs[r, col_pred], Xg, Yg, Zp, "jet",
                    f"{name}  {lab}  (t={t})",
                    vmin=vmin, vmax=vmax, levels=levels_main)

                axs[r, col_err].set_xlim(*xlim_t)
                axs[r, col_err].set_ylim(*ylim_t)
                _cf(axs[r, col_err], Xg, Yg, Ze, "inferno",
                    f"|Err|  {name}  {lab}  (t={t})",
                    levels=levels_err)

        row_base += nF

    # ── optional file save ─────────────────────
    if save_files:
        if out_csv and all_df:
            combined = pd.concat(all_df, ignore_index=True)
            combined.to_csv(out_csv, index=False)

        if out_fig:
            fig.savefig(out_fig, dpi=dpi, facecolor="white",
                        bbox_inches="tight")

    return fig, (pd.concat(all_df, ignore_index=True) if all_df else None)


# ─────────────────────────────────────────────
# PER-VARIABLE BAR CHARTS (white)
# ─────────────────────────────────────────────
def plot_performance_bars(metrics_df, metric_key="RMSE"):
    variables = metrics_df["Variable"].unique()
    n_vars    = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(4.5 * n_vars, 4),
                             constrained_layout=True)
    fig.patch.set_facecolor("white")
    if n_vars == 1:
        axes = [axes]

    palette = ["#222222", "#888888", "#cccccc"]

    for ax, var in zip(axes, variables):
        ax.set_facecolor("white")
        sub     = metrics_df[metrics_df["Variable"] == var]
        models  = sub["Model"].tolist()
        vals    = sub[metric_key].tolist()
        colors  = [palette[i % len(palette)] for i in range(len(models))]
        bars    = ax.bar(models, vals, color=colors, width=0.55, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{val:.5f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#111111")
        ax.set_title(f"{metric_key}  —  {var}", fontsize=11,
                     fontweight="bold", color="#111111")
        ax.set_ylabel(metric_key, fontsize=9, color="#333333")
        ax.tick_params(colors="#333333", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
        ax.set_xticklabels(models, fontsize=9, color="#111111")
    return fig


def plot_r2_grouped(metrics_df):
    variables   = metrics_df["Variable"].unique().tolist()
    model_names = metrics_df["Model"].unique().tolist()
    x           = np.arange(len(variables))
    width       = 0.22
    palette     = ["#222222", "#888888", "#cccccc"]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, mname in enumerate(model_names):
        sub  = metrics_df[metrics_df["Model"] == mname]
        r2s  = []
        for var in variables:
            row = sub[sub["Variable"] == var]
            r2s.append(float(row["R2 Score"].values[0]) if len(row) else 0.0)
        offset = (i - (len(model_names) - 1) / 2) * width
        ax.bar(x + offset, r2s, width, label=mname,
               color=palette[i % len(palette)], edgecolor="white", alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=10, color="#111111")
    ax.set_ylabel("R2 Score", fontsize=10, color="#333333")
    ax.set_title("R2 Score Comparison — All Variables", fontsize=12,
                 fontweight="bold", color="#111111")
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
              labelcolor="#111111")
    ax.axhline(1.0, color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors="#333333")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    return fig


def plot_error_dist(out_true, preds_dict, var_idx=0, var_name="U"):
    fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    palette = ["#222222", "#888888", "#aaaaaa"]
    for i, (mname, pred) in enumerate(preds_dict.items()):
        err = out_true[:, var_idx] - pred[:, var_idx]
        mask = ~np.isnan(err)
        ax.hist(err[mask], bins=80, alpha=0.7, label=mname,
                color=palette[i % len(palette)], edgecolor="none")
    ax.axvline(0, color="#cc0000", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Prediction Error", fontsize=10, color="#333333")
    ax.set_ylabel("Count", fontsize=10, color="#333333")
    ax.set_title(f"Error Distribution — {var_name}", fontsize=11,
                 fontweight="bold", color="#111111")
    ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
              labelcolor="#111111")
    ax.tick_params(colors="#333333")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    return fig


def plot_scatter_pred_vs_true(out_true, preds_dict, var_idx=0, var_name="U"):
    n_m  = len(preds_dict)
    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4.5),
                             constrained_layout=True)
    fig.patch.set_facecolor("white")
    if n_m == 1:
        axes = [axes]
    palette = ["#222222", "#888888", "#aaaaaa"]

    for i, (mname, pred) in enumerate(preds_dict.items()):
        ax        = axes[i]
        ax.set_facecolor("white")
        yt        = out_true[:, var_idx]
        yp        = pred[:, var_idx]
        mask      = ~np.isnan(yt) & ~np.isnan(yp)
        yt, yp    = yt[mask], yp[mask]
        ax.scatter(yt, yp, s=3, alpha=0.45, color=palette[i % len(palette)])
        lo, hi    = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], color="#cc0000", linewidth=1, linestyle="--")
        r2        = 1 - np.sum((yt - yp)**2) / (np.sum((yt - np.mean(yt))**2) + 1e-12)
        ax.set_title(f"{mname}  {var_name}  |  R2={r2:.4f}", fontsize=9,
                     fontweight="bold", color="#111111")
        ax.set_xlabel("CFD (True)", fontsize=9, color="#333333")
        ax.set_ylabel("Predicted",  fontsize=9, color="#333333")
        ax.tick_params(colors="#333333", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
    return fig


# ─────────────────────────────────────────────
# FLUID FLOW NARRATIVE SUMMARY
# ─────────────────────────────────────────────
def generate_flow_summary(out_true, preds_dict, timestamp):
    U  = out_true[:, 0]
    V  = out_true[:, 1]
    W  = out_true[:, 2]

    valid_U = U[~np.isnan(U)]
    valid_V = V[~np.isnan(V)]
    valid_W = W[~np.isnan(W)]

    if len(valid_U) == 0:
        return "Ground truth data not available for flow summary."

    speed_mag     = np.sqrt(valid_U**2 + valid_V**2 + valid_W**2)
    recirculation = np.sum(valid_U < 0) / len(valid_U) * 100
    hotspot_frac  = np.sum(speed_mag > np.percentile(speed_mag, 95)) / len(speed_mag) * 100
    asym_ratio    = np.abs(np.mean(valid_V)) / (np.mean(np.abs(valid_V)) + 1e-10)
    dom_dir       = "axial (Z/W)" if np.std(valid_W) > np.std(valid_U) \
                    else "streamwise (X/U)" if np.std(valid_U) > np.std(valid_V) \
                    else "transverse (Y/V)"

    model_lines = []
    best_model, best_rmse = None, 1e10
    for mname, pred in preds_dict.items():
        mask = ~np.isnan(out_true[:, 0])
        rmse = np.sqrt(np.mean((out_true[mask] - pred[mask])**2))
        r2   = 1 - np.sum((out_true[mask] - pred[mask])**2) / \
                   (np.sum((out_true[mask] - np.mean(out_true[mask]))**2) + 1e-12)
        model_lines.append(f"- **{mname}**: RMSE = `{rmse:.5f}`, R2 = `{r2:.4f}`")
        if rmse < best_rmse:
            best_rmse, best_model = rmse, mname

    model_summary = "\n".join(model_lines)

    return f"""
**Timestamp Analyzed:** `t = {timestamp}`

---

### What Is Actually Happening Inside This Microchannel

The fluid **does not flow uniformly** — it behaves as a spatially organized, nonlinear system:

- **Dominant flow direction: {dom_dir}.** Most of the kinetic energy is concentrated along this axis. The other velocity components are **not zero** — they encode secondary circulations driven by the non-trivial channel geometry.

- **{recirculation:.1f}% of spatial points** show negative streamwise (U) velocity. These are genuine **recirculation zones** — pockets where fluid reverses direction, typically forming downstream of curved walls or geometric constrictions. They locally reduce effective mass flow rate but significantly enhance mixing.

- **Lateral asymmetry index: {asym_ratio:.3f}.** When this value exceeds 0.3 the channel geometry is forcing the fluid core to drift off its centerline — a hallmark of **Dean-type vortex pairs** in curved or non-symmetric microchannels. The flow is not symmetric, even if the inlet boundary condition was.

- **The top 5% high-speed locations** account for `{hotspot_frac:.1f}%` of the total flow energy. These are **jet-like corridors** — narrow streaks where continuity forces acceleration through constricted cross-sections. They are the primary mass and heat transport pathways.

- **W velocity range: `{valid_W.min():.4f}` to `{valid_W.max():.4f}`.** Non-trivial out-of-plane velocity in a nominally 2D cross-section reveals **3D helical flow structures** — the fluid is corkscrewing through the channel. This dramatically improves heat/mass transfer at the cost of a pressure penalty.

---

### Neural Model Performance

{model_summary}

**Best model: `{best_model}`** with RMSE = `{best_rmse:.5f}`

The error maps (inferno colormap) localize exactly **where** each model fails:
1. **Boundary-layer regions** — steep gradients near walls require very high spatial resolution to learn accurately.
2. **Recirculation cores** — highly nonlinear; small input perturbations cause large output swings.
3. **High-W out-of-plane zones** — the hardest to predict because 3D pressure balance is not directly encoded in the 2D XY input space.

---

### Practical Interpretation

If this microchannel is used for **chip-scale thermal management**, the recirculation zones and helical flow patterns are **thermally beneficial** — they break the stagnant thermal boundary layer and promote mixing. However, they also impose **non-uniform wall shear stress**, which matters for long-term structural fatigue and biofouling in lab-on-chip applications.
The high-velocity corridors (top 5%) should be spatially aligned with the **device hotspot locations** to maximize cooling effectiveness.
"""


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## Model Paths")
st.sidebar.markdown("---")

unetAM_folder = st.sidebar.text_input("U-Net AM (GELU) folder", value="saved_models/unetAM")
tnet_folder   = st.sidebar.text_input("T-Net (ReLU) folder",    value="saved_models/tnet")
unet_folder   = st.sidebar.text_input("U-Net (Tanh) folder",    value="saved_models/unet_tanh")

unetAM_name   = st.sidebar.text_input("U-Net AM model name",    value="model_unetAM_gelu")
tnet_name     = st.sidebar.text_input("T-Net model name",       value="model_tnet_relu")
unet_name     = st.sidebar.text_input("U-Net Tanh model name",  value="model_unet_tanh")

st.sidebar.markdown("---")
st.sidebar.markdown("## Plot Settings")

time_levels_str = st.sidebar.text_input(
    "Timesteps to plot (comma-separated)", value="10, 15, 20, 40"
)
levels_main = st.sidebar.slider("Contour levels (prediction)", 50, 300, 150, step=10)
levels_err  = st.sidebar.slider("Contour levels (error)",      50, 300, 150, step=10)
xlim_lo     = st.sidebar.number_input("X-axis min", value=-1.1)
xlim_hi     = st.sidebar.number_input("X-axis max", value=1.1)
ylim_lo     = st.sidebar.number_input("Y-axis min", value=-1.1)
ylim_hi     = st.sidebar.number_input("Y-axis max", value=1.1)
row_height  = st.sidebar.slider("Row height (in)", 2.0, 6.0, 3.2, step=0.2)
col_width   = st.sidebar.slider("Col width (in)",  2.5, 7.0, 4.4, step=0.2)
save_files  = st.sidebar.checkbox("Save figure + CSV to disk", value=False)
out_fig     = st.sidebar.text_input("Output figure filename", value="comparison_plot.tiff")
out_csv     = st.sidebar.text_input("Output CSV filename",    value="predictions.csv")

st.sidebar.markdown("---")
st.sidebar.caption("Upload a .dat file and click Run Analysis.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("# Microchannel Flow Neural Operator")
st.markdown("##### CFD vs. Neural Network Prediction — Multi-Timestep Analysis")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a .dat CFD file  (expected name: flow_data_<timestamp>.dat)",
    type=["dat", "txt", "csv"],
)

run_btn = st.button("Run Analysis")

if uploaded_file is not None and run_btn:

    # ── parse timestep from filename ─────────
    timestamp = extract_timestamp(uploaded_file.name)

    # ── parse data file ───────────────────────
    with st.spinner("Parsing .dat file..."):
        df_raw, err_msg = parse_dat_file(uploaded_file)

    if df_raw is None:
        st.error(f"Failed to parse file: {err_msg}")
        st.stop()

    df_raw["T"] = timestamp
    xyt_raw     = build_xyt_array(df_raw, timestamp)
    out_true    = build_out_array(df_raw)

    st.success(
        f"Loaded **{len(df_raw):,}** points from `{uploaded_file.name}` — "
        f"Timestamp: `t = {timestamp}`"
    )

    # ── load models ───────────────────────────
    model_configs = [
        ("U-Net AM", unetAM_folder, unetAM_name, "gelu"),
        ("T-Net",    tnet_folder,   tnet_name,   "relu"),
        ("U-Net",    unet_folder,   unet_name,   "tanh"),
    ]
    models_loaded = {}

    for label, folder, mname, act in model_configs:
        if not os.path.exists(folder):
            st.warning(f"Folder not found: `{folder}` — skipping `{label}`.")
            continue
        try:
            with st.spinner(f"Loading {label}..."):
                model, xyt_sc, out_sc = load_model_from_disk(folder, mname, act)
            models_loaded[label] = (model, xyt_sc, out_sc)
            st.success(f"`{label}` loaded.")
        except Exception as e:
            st.error(f"Could not load `{label}`: {e}")

    if not models_loaded:
        st.error("No models loaded. Check folder paths and model names in the sidebar.")
        st.stop()

    # ── predictions ───────────────────────────
    with st.spinner("Running predictions for all models..."):
        preds_dict = predict_all_models(models_loaded, xyt_raw)

    st.success("Predictions complete.")

    # ── parse timestep list ───────────────────
    try:
        time_levels = tuple(int(x.strip()) for x in time_levels_str.split(","))
    except ValueError:
        st.error("Invalid timestep list. Use integers separated by commas.")
        st.stop()

    # ── METRIC CARDS ──────────────────────────
    st.markdown("---")
    st.markdown("### Overall Model Performance")
    cols = st.columns(len(preds_dict))
    for i, (mname, pred) in enumerate(preds_dict.items()):
        mask = ~np.isnan(out_true[:, 0])
        rmse = np.sqrt(mean_squared_error(out_true[mask], pred[mask]))
        mae  = mean_absolute_error(out_true[mask], pred[mask])
        r2   = r2_score(out_true[mask], pred[mask])
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{rmse:.5f}</div>
                <div class="metric-lbl">RMSE</div>
                <br>
                <div class="metric-val">{r2:.4f}</div>
                <div class="metric-lbl">R2 Score</div>
                <br>
                <div class="metric-val">{mae:.5f}</div>
                <div class="metric-lbl">MAE</div>
                <br>
                <span class="tag">{mname}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── MAIN COMPARISON FIGURE ────────────────
    st.markdown("---")
    st.markdown("### Contour Comparison — CFD | Prediction | Absolute Error")
    st.caption(
        "Rows: each variable (U, V, W) repeated per timestep  |  "
        "Columns: CFD (t=T) | Model Pred | |Error|  ... per model"
    )

    with st.spinner("Rendering main comparison figure (this may take a moment)..."):
        fig_main, df_combined = plot_inline_comparison(
            xyt_raw      = xyt_raw,
            out_true     = out_true,
            preds_dict   = preds_dict,
            time_levels  = time_levels,
            labels       = ("U", "V", "W"),
            levels_main  = levels_main,
            levels_err   = levels_err,
            xlim         = (xlim_lo, xlim_hi),
            ylim         = (ylim_lo, ylim_hi),
            row_height   = row_height,
            col_width    = col_width,
            dpi          = 150,
            out_fig      = out_fig if save_files else None,
            out_csv      = out_csv if save_files else None,
            save_files   = save_files,
        )
        st.pyplot(fig_main, use_container_width=True)
    plt.close(fig_main)

    # ── PER-VARIABLE DETAILED ANALYSIS ────────
    st.markdown("---")
    st.markdown("### Per-Variable Detailed Analysis")

    var_names = ["U", "V", "W", "Um"]
    tabs      = st.tabs([f"Variable: {v}" for v in var_names])

    metrics_df_list = []

    for vidx, (vname, tab) in enumerate(zip(var_names, tabs)):
        with tab:

            # metrics table
            rows = []
            for mname, pred in preds_dict.items():
                yt   = out_true[:, vidx]
                yp   = pred[:, vidx]
                mask = ~np.isnan(yt) & ~np.isnan(yp)
                if mask.sum() == 0:
                    continue
                yt, yp = yt[mask], yp[mask]
                mse  = np.mean((yt - yp) ** 2)
                mae  = np.mean(np.abs(yt - yp))
                rmse = np.sqrt(mse)
                maxae = np.max(np.abs(yt - yp))
                r2   = 1 - np.sum((yt - yp)**2) / (np.sum((yt - np.mean(yt))**2) + 1e-12)
                rel_e = np.mean(np.abs(yt - yp) / (np.abs(yt) + 1e-12)) * 100
                row = {
                    "Variable": vname, "Model": mname,
                    "MSE": mse, "MAE": mae, "RMSE": rmse,
                    "Max AE": maxae, "R2 Score": r2, "Rel Error (%)": rel_e,
                }
                rows.append(row)
                metrics_df_list.append(row)

            if rows:
                df_var = pd.DataFrame(rows).set_index("Model")
                st.dataframe(
                    df_var.drop(columns=["Variable"])
                          .style
                          .highlight_min(axis=0, color="#e8f5e9")
                          .format("{:.6f}"),
                    use_container_width=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    fig_b = plot_performance_bars(pd.DataFrame(rows), metric_key="RMSE")
                    st.pyplot(fig_b, use_container_width=True)
                    plt.close(fig_b)
                with c2:
                    fig_b2 = plot_performance_bars(pd.DataFrame(rows), metric_key="MAE")
                    st.pyplot(fig_b2, use_container_width=True)
                    plt.close(fig_b2)

                fig_sc = plot_scatter_pred_vs_true(out_true, preds_dict,
                                                   var_idx=vidx, var_name=vname)
                st.pyplot(fig_sc, use_container_width=True)
                plt.close(fig_sc)

                fig_ed = plot_error_dist(out_true, preds_dict,
                                         var_idx=vidx, var_name=vname)
                st.pyplot(fig_ed, use_container_width=True)
                plt.close(fig_ed)
            else:
                st.info(f"No ground truth available for {vname}.")

    # ── R2 GROUPED BAR ────────────────────────
    if metrics_df_list:
        st.markdown("---")
        st.markdown("### R2 Score — All Variables and Models")
        df_all_metrics = pd.DataFrame(metrics_df_list)
        fig_r2 = plot_r2_grouped(df_all_metrics)
        st.pyplot(fig_r2, use_container_width=True)
        plt.close(fig_r2)

        # full metrics export table
        st.markdown("---")
        st.markdown("### Full Metrics Table")
        st.dataframe(
            df_all_metrics.style.format({
                "MSE": "{:.6f}", "MAE": "{:.6f}", "RMSE": "{:.6f}",
                "Max AE": "{:.6f}", "R2 Score": "{:.4f}", "Rel Error (%)": "{:.3f}",
            }),
            use_container_width=True,
        )

        csv_bytes = df_all_metrics.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Metrics CSV",
            data=csv_bytes,
            file_name=f"metrics_t{timestamp}.csv",
            mime="text/csv",
        )

    # ── FLOW SUMMARY ──────────────────────────
    st.markdown("---")
    st.markdown("### Fluid Flow Pattern Summary")
    st.caption("Non-intuitive, plain-language interpretation of what the CFD data reveals")

    summary = generate_flow_summary(out_true, preds_dict, timestamp)
    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

    # ── DOWNLOAD PREDICTION FIGURE ────────────
    if df_combined is not None:
        st.markdown("---")
        st.markdown("### Download Prediction Data")
        pred_csv = df_combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction CSV (all timesteps)",
            data=pred_csv,
            file_name=out_csv if save_files else "predictions.csv",
            mime="text/csv",
        )

elif not run_btn:
    st.info("Upload a `.dat` file and click **Run Analysis** to begin.")
