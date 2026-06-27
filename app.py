# -*- coding: utf-8 -*-
"""
==========================================================================================
STREAMLIT WEB APP: TỐI ƯU HÓA DANH MỤC ĐẦU TƯ TRÊN HOSE 2020-2023
Chiến lược: Bollinger Bands + RSI + Volume Filter kết hợp Tối ưu hóa bầy đàn (PSO)
==========================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Vẽ biểu đồ nâng cao
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kiểm định thống kê
try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ==========================================================================================
# 0. CẤU HÌNH TRANG & CSS TÙY CHỈNH
# ==========================================================================================
st.set_page_config(
    page_title="Tối Ưu Hóa Danh Mục Đầu Tư HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS tùy chỉnh để ứng dụng nhìn hiện đại và premium hơn
st.markdown("""
<style>
    /* Nhập font Outfit từ Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Cấu hình tiêu đề */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding-top: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Thiết kế Metrics Cards đẹp mắt */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #495057;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1f4068;
        margin-top: 0.3rem;
    }
    
    .metric-sub {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    
    /* Căn chỉnh thông tin thông báo */
    .highlight-box {
        background-color: #eef1f6;
        border-left: 5px solid #1f4068;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Các giới hạn tham số của chiến lược để PSO tìm kiếm
PARAM_BOUNDS = [
    ("bb_window",   10,   40,  True),   # Chu kỳ SMA của Bollinger Bands
    ("bb_std",       1.5,  3.0, False), # Độ lệch chuẩn
    ("rsi_window",   7,    25,  True),  # Chu kỳ RSI
    ("vol_window",   10,   40,  True),  # Chu kỳ Volume SMA
    ("lower",       20,   45,  False),  # Ngưỡng RSI quá bán (mua)
    ("upper",       55,   85,  False),  # Ngưỡng RSI quá mua (bán)
]

# Khởi tạo session_state để lưu trữ tham số tối ưu và các trạng thái tính toán
if "best_p" not in st.session_state:
    # Tham số mặc định của hệ thống
    st.session_state.best_p = {
        "bb_window": 20,
        "bb_std": 2.0,
        "rsi_window": 14,
        "vol_window": 20,
        "lower": 30.0,
        "upper": 70.0
    }

if "top5" not in st.session_state:
    st.session_state.top5 = []

if "rank_df" not in st.session_state:
    st.session_state.rank_df = None

if "is_ret" not in st.session_state:
    st.session_state.is_ret = {}

# ==========================================================================================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU (CACHED)
# ==========================================================================================
@st.cache_data(show_spinner="Đang xử lý và tải dữ liệu...")
def load_and_prepare_data(csv_path_or_file):
    df = pd.read_csv(csv_path_or_file, low_memory=False)
    # Loại bỏ các cột Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Xử lý ngày tháng
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.sort_values(["ticker", "date"])

    # Tạo các ma trận Pivot để tính toán vectorized
    close = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    open_ = df.pivot(index="date", columns="ticker", values="adj_open").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").sort_index()

    # Xử lý dữ liệu trống (Fill forward cho giá đóng/mở, 0 cho volume)
    close = close.ffill()
    open_ = open_.ffill()
    volume = volume.fillna(0.0)

    # Lấy danh sách cổ phiếu hợp lệ (bỏ VNINDEX)
    stocks = [c for c in close.columns if c != "VNINDEX"]
    
    return close, open_, volume, stocks, df

# ==========================================================================================
# 2. HÀM TÍNH TOÁN RSI DẠNG MA TRẬN VECTORIZED
# ==========================================================================================
def rsi_matrix(close_df, window):
    delta = close_df.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.rolling(window=window, min_periods=1).mean()
    al = loss.rolling(window=window, min_periods=1).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

# ==========================================================================================
# 3. TẠO TÍN HIỆU CHIẾN LƯỢC BB + RSI + VOLUME
# ==========================================================================================
def bbrsi_vol_signals_matrix(close_df, vol_df, p):
    """Tính toán tín hiệu Mua/Bán cho toàn bộ danh mục cùng lúc dạng ma trận."""
    # 1. Tính Bollinger Bands
    sma = close_df.rolling(window=int(p["bb_window"]), min_periods=1).mean()
    std = close_df.rolling(window=int(p["bb_window"]), min_periods=1).std()
    lower_band = sma - p["bb_std"] * std
    upper_band = sma + p["bb_std"] * std

    # 2. Tính RSI
    rsi = rsi_matrix(close_df, int(p["rsi_window"]))

    # 3. Tính Volume trung bình
    vol_sma = vol_df.rolling(window=int(p["vol_window"]), min_periods=1).mean()

    # 4. Logic Buy / Sell
    buy  = (close_df <= lower_band) & (rsi < p["lower"]) & (vol_df > vol_sma)
    sell = (close_df >= upper_band) & (rsi > p["upper"])

    pos  = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    # Gán 1.0 khi Mua, 0.0 khi Bán, sau đó ffill để nắm giữ vị thế
    pos  = pos.mask(buy, 1.0).mask(sell, 0.0).ffill().fillna(0.0)
    return pos

def bbrsi_vol_signals_single(close_series, vol_series, p):
    """Tính toán tín hiệu cho MỘT cổ phiếu riêng lẻ."""
    sma = close_series.rolling(window=int(p["bb_window"]), min_periods=1).mean()
    std = close_series.rolling(window=int(p["bb_window"]), min_periods=1).std()
    lower_band = sma - p["bb_std"] * std
    upper_band = sma + p["bb_std"] * std

    delta = close_series.diff()
    ag = delta.clip(lower=0).rolling(window=int(p["rsi_window"]), min_periods=1).mean()
    al = -delta.clip(upper=0).rolling(window=int(p["rsi_window"]), min_periods=1).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    vol_sma = vol_series.rolling(window=int(p["vol_window"]), min_periods=1).mean()

    buy  = (close_series <= lower_band) & (rsi < p["lower"]) & (vol_series > vol_sma)
    sell = (close_series >= upper_band) & (rsi > p["upper"])

    pos  = pd.Series(np.nan, index=close_series.index)
    pos  = pos.mask(buy, 1.0).mask(sell, 0.0).ffill().fillna(0.0)
    return pos, buy, sell

# ==========================================================================================
# 4. BACKTEST ENGINES (Numpy Vectorized)
# ==========================================================================================
def backtest_single(open_arr, close_arr, desired_pos, capital, fee):
    n = len(close_arr)
    hold = np.zeros(n)
    hold[1:] = desired_pos[:-1]
    prev = np.zeros(n); prev[1:] = hold[:-1]
    o, c = open_arr, close_arr
    Cprev = np.roll(c, 1)
    
    enter = (hold == 1) & (prev == 0)
    both  = (hold == 1) & (prev == 1)
    exit_ = (hold == 0) & (prev == 1)
    
    factor = np.ones(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor[enter] = (c[enter] / o[enter]) * (1 - fee)
        factor[both]  = c[both] / Cprev[both]
        factor[exit_] = (o[exit_] / Cprev[exit_]) * (1 - fee)
    factor[0] = 1.0
    equity = capital * np.cumprod(factor)
    n_trades = int(enter.sum() + exit_.sum())
    return equity, n_trades

def backtest_matrix(open_df, close_df, dp_df, capital, fee):
    O, C, DP = open_df.values, close_df.values, dp_df.values
    hold = np.zeros_like(C); hold[1:] = DP[:-1]
    prev = np.zeros_like(C); prev[1:] = hold[:-1]
    Cprev = np.roll(C, 1, axis=0)
    
    enter = (hold == 1) & (prev == 0)
    both  = (hold == 1) & (prev == 1)
    exit_ = (hold == 0) & (prev == 1)
    
    factor = np.ones_like(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(enter, C / O * (1 - fee), factor)
        factor = np.where(both,  C / Cprev,         factor)
        factor = np.where(exit_, O / Cprev * (1 - fee), factor)
    factor[0, :] = 1.0
    equity = capital * np.cumprod(factor, axis=0)
    ntr = (enter.sum(axis=0) + exit_.sum(axis=0)).astype(int)
    return equity, ntr

# ==========================================================================================
# 5. TÍNH CHỈ SỐ HIỆU QUẢ VÀ SHARPE VECTOR
# ==========================================================================================
def perf_metrics(equity, n_trades=None, initial=1_000_000_000, rf_daily=0.04/252, periods=252):
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return {
            "Total Return [%]": 0.0, "CAGR [%]": 0.0, "Volatility [%]": 0.0,
            "Sharpe": 0.0, "Sortino": 0.0, "Max Drawdown [%]": 0.0, "Calmar": 0.0, "Final Value": initial
        }
    
    ret = equity[1:] / equity[:-1] - 1.0
    total_return = equity[-1] / initial - 1.0
    n = len(equity)
    years = n / periods
    
    cagr = (equity[-1] / initial) ** (1 / years) - 1.0 if equity[-1] > 0 else -1.0
    vol  = ret.std(ddof=1) * np.sqrt(periods) if len(ret) > 1 else np.nan
    excess = ret - rf_daily
    sd = ret.std(ddof=1)
    
    sharpe = np.sqrt(periods) * excess.mean() / sd if sd > 0 else np.nan
    downside = np.minimum(excess, 0.0)
    dd_dev = np.sqrt((downside ** 2).mean())
    sortino = np.sqrt(periods) * excess.mean() / dd_dev if dd_dev > 0 else np.nan
    
    run_max = np.maximum.accumulate(equity)
    mdd = (equity / run_max - 1.0).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    
    out = {
        "Total Return [%]": total_return * 100,
        "CAGR [%]": cagr * 100,
        "Volatility [%]": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown [%]": mdd * 100,
        "Calmar": calmar,
        "Final Value": equity[-1],
    }
    if n_trades is not None:
        out["Trades"] = n_trades
    return out

def sharpe_vector(equity, rf_daily=0.04/252, periods=252):
    ret = equity[1:] / equity[:-1] - 1.0
    excess = ret - rf_daily
    sd = ret.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(periods) * excess.mean(axis=0) / sd

# ==========================================================================================
# 6. TỐI ƯU HÓA BẦY ĐÀN (PSO)
# ==========================================================================================
def decode(vec):
    p = {}
    for val, (name, lo, hi, is_int) in zip(vec, PARAM_BOUNDS):
        v = lo + (hi - lo) * val
        p[name] = int(round(v)) if is_int else float(v)
    if p["upper"] <= p["lower"]: 
        p["upper"] = p["lower"] + 5.0
    return p

def pso_optimize(fitness_fn, n_particles=15, n_iter=15, seed=42,
                 w=0.7, c1=1.5, c2=1.5, progress_bar=None, status_text=None):
    rng = np.random.default_rng(seed)
    dim = len(PARAM_BOUNDS)
    X = rng.random((n_particles, dim))
    V = rng.uniform(-0.1, 0.1, (n_particles, dim))
    
    pbest = X.copy()
    pbest_val = np.array([fitness_fn(decode(x)) for x in X])
    
    g = int(np.argmax(pbest_val))
    gbest = pbest[g].copy()
    gbest_val = pbest_val[g]
    
    for it in range(n_iter):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -0.3, 0.3)
        X = np.clip(X + V, 0.0, 1.0)
        
        vals = np.array([fitness_fn(decode(x)) for x in X])
        imp = vals > pbest_val
        pbest[imp] = X[imp]
        pbest_val[imp] = vals[imp]
        
        if pbest_val.max() > gbest_val:
            g = int(np.argmax(pbest_val))
            gbest = pbest[g].copy()
            gbest_val = pbest_val[g]
            
        # Cập nhật tiến trình trên Streamlit
        if progress_bar is not None:
            progress_bar.progress((it + 1) / n_iter)
        if status_text is not None:
            status_text.text(f"PSO Vòng {it+1}/{n_iter}: Sharpe In-Sample trung bình tốt nhất = {gbest_val:.4f}")
            
    return decode(gbest), gbest_val

# ==========================================================================================
# 7. ENGINE DANH MỤC (Phân bổ tỷ trọng & Tái cân bằng định kỳ)
# ==========================================================================================
def rebalance_indices(dates, freq, oos_start_pos):
    if freq == "none":
        return set()
    idx = []
    for pos in range(oos_start_pos + 1, len(dates)):
        d, prev = dates[pos], dates[pos - 1]
        if freq == "monthly" and (d.year, d.month) != (prev.year, prev.month):
            idx.append(pos)
        elif freq == "quarterly" and (d.year, (d.month - 1)//3) != (prev.year, (prev.month - 1)//3):
            idx.append(pos)
        elif freq == "annual" and d.year != prev.year:
            idx.append(pos)
    return set(idx)

def compute_weights(scheme, perf_vector, n):
    if scheme == "equal":
        return np.full(n, 1.0 / n)
    pos = np.clip(perf_vector, 0.0, None)
    return np.full(n, 1.0 / n) if pos.sum() <= 0 else pos / pos.sum()

def run_portfolio(close_df, open_df, vol_df, stocks, params, weight_scheme, rebalance,
                  init_perf, capital, fee, oos_start_date, rf_daily, periods):
    dates = close_df.index
    N = len(stocks)
    
    # Tính toán tín hiệu cho rổ cổ phiếu được chọn
    dp = bbrsi_vol_signals_matrix(close_df[stocks], vol_df[stocks], params).values
    C  = close_df[stocks].values
    O  = open_df[stocks].values
    
    # Tìm điểm bắt đầu Out-of-Sample
    try:
        oos_pos = list(dates).index(pd.Timestamp(oos_start_date))
    except ValueError:
        # Nếu không khớp chính xác, tìm ngày gần nhất tiếp theo
        oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start_date))
        
    rebal = rebalance_indices(dates, rebalance, oos_pos)

    # Thiết lập tỷ trọng ban đầu
    init_perf_vec = np.array([init_perf.get(tk, 1.0) for tk in stocks], dtype=float)
    w0 = compute_weights(weight_scheme, init_perf_vec, N)
    
    cash = w0 * capital
    shares = np.zeros(N)
    inpos = np.zeros(N, dtype=bool)
    last_rebal_value = w0 * capital

    eq_dates, eq_vals = [], []
    total_trades = 0
    total_fee_paid = 0.0

    # Chạy vòng lặp mô phỏng hàng ngày
    for pos in range(oos_pos, len(dates)):
        o_t, c_t = O[pos], C[pos]
        regime_prev = dp[pos - 1]

        # Kiểm tra lệnh Mua / Bán hàng ngày
        for i in range(N):
            if regime_prev[i] == 1 and not inpos[i]:
                if cash[i] > 0:
                    f = cash[i] * fee
                    shares[i] = (cash[i] - f) / o_t[i]
                    total_fee_paid += f
                    cash[i] = 0.0
                    total_trades += 1
                inpos[i] = True
            elif regime_prev[i] == 0 and inpos[i]:
                proceeds = shares[i] * o_t[i]
                f = proceeds * fee
                cash[i] = proceeds - f
                total_fee_paid += f
                shares[i] = 0.0
                inpos[i] = False
                total_trades += 1

        # Thực hiện Tái Cân Bằng (Rebalancing) định kỳ
        if pos in rebal:
            sleeve_val = cash + shares * c_t
            V = sleeve_val.sum()
            if weight_scheme == "performance":
                growth = sleeve_val / np.where(last_rebal_value > 0, last_rebal_value, np.nan) - 1.0
                growth = np.nan_to_num(growth, nan=0.0)
                w = compute_weights("performance", growth, N)
            else:
                w = compute_weights("equal", None, N)
                
            tgt = w * V
            reb_fee = sum(fee * abs(tgt[i] - shares[i] * c_t[i]) for i in range(N) if inpos[i])
            scale = (V - reb_fee) / V if V > 0 else 1.0
            
            for i in range(N):
                if inpos[i]:
                    shares[i] = (tgt[i] * scale) / c_t[i]
                    cash[i] = 0.0
                else:
                    cash[i] = tgt[i] * scale
                    shares[i] = 0.0
            total_fee_paid += reb_fee
            last_rebal_value = tgt * scale

        eq_dates.append(dates[pos])
        eq_vals.append(cash.sum() + (shares * c_t).sum())

    equity = pd.Series(eq_vals, index=eq_dates)
    m = perf_metrics(equity.values, total_trades, initial=capital, rf_daily=rf_daily, periods=periods)
    m["Fees Paid"] = total_fee_paid
    return equity, m

# ==========================================================================================
# 8. CÁC PHƯƠNG PHÁP THỰ ĐỘNG (BENCHMARKS)
# ==========================================================================================
def buy_hold_basket(close_df, open_df, stocks, weights=None, capital=1_000_000_000, fee=0.0015, oos_start_date="2021-01-04", rf_daily=0.04/252, periods=252):
    dates = close_df.index
    try:
        oos_pos = list(dates).index(pd.Timestamp(oos_start_date))
    except ValueError:
        oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start_date))
        
    N = len(stocks)
    if weights is None: 
        weights = np.full(N, 1.0 / N)
        
    o0 = open_df[stocks].values[oos_pos]
    shares = (weights * capital * (1 - fee)) / o0
    C = close_df[stocks].values[oos_pos:]
    equity = pd.Series((C * shares).sum(axis=1), index=dates[oos_pos:])
    return equity, perf_metrics(equity.values, 1, initial=capital, rf_daily=rf_daily, periods=periods)

def buy_hold_index(close_df, ticker="VNINDEX", capital=1_000_000_000, oos_start_date="2021-01-04", rf_daily=0.04/252, periods=252):
    dates = close_df.index
    try:
        oos_pos = list(dates).index(pd.Timestamp(oos_start_date))
    except ValueError:
        oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start_date))
        
    s = close_df[ticker].values[oos_pos:]
    equity = pd.Series(capital * s / s[0], index=dates[oos_pos:])
    return equity, perf_metrics(equity.values, 0, initial=capital, rf_daily=rf_daily, periods=periods)

# ==========================================================================================
# 9. PHÂN TÍCH THEO NĂM
# ==========================================================================================
def submetrics(equity, label, initial_capital, rf_daily, periods):
    out = {"label": label, "full": perf_metrics(equity.values, initial=initial_capital, rf_daily=rf_daily, periods=periods)}
    years = sorted(list(set(equity.index.year)))
    for yr in years:
        seg = equity[equity.index.year == yr]
        prev = equity[equity.index.year < yr]
        start_val = prev.values[-1] if len(prev) else initial_capital
        e = np.concatenate([[start_val], seg.values])
        out[str(yr)] = perf_metrics(e, initial=start_val, rf_daily=rf_daily, periods=periods)
    return out

# ==========================================================================================
# 10. GIAO DIỆN STREAMLIT CHÍNH
# ==========================================================================================
st.markdown('<div class="main-title">📈 Tối Ưu Hóa Danh Mục Đầu Tư HOSE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Web App tối ưu danh mục cổ phiếu bằng chiến lược Bollinger Bands, RSI, Volume kết hợp thuật toán tối ưu bầy đàn (PSO)</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.header("📁 Dữ Liệu Đầu Vào")

# Sử dụng dữ liệu có sẵn nếu tồn tại
default_csv = "HOSE_2020_2023.csv"
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu sàn HOSE (Nếu muốn sử dụng dữ liệu mới)", type=["csv"])

data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
    st.sidebar.success("Đã nhận file tải lên!")
elif os.path.exists(default_csv):
    data_source = default_csv
    st.sidebar.info("Đang sử dụng dữ liệu mặc định: `HOSE_2020_2023.csv`")
else:
    st.sidebar.error("Không tìm thấy file dữ liệu mặc định! Vui lòng tải lên file CSV.")
    st.stop()

# Đọc dữ liệu
close, open_, volume, STOCKS, df_raw = load_and_prepare_data(data_source)

# THAM SỐ GIAO DỊCH CHUNG
st.sidebar.header("⚙️ Tham Số Hệ Thống")
capital = st.sidebar.number_input("Vốn đầu tư ban đầu (VND)", min_value=10_000_000, max_value=100_000_000_000, value=1_000_000_000, step=100_000_000, format="%d")
rf_annual = st.sidebar.slider("Lãi suất phi rủi ro năm (%)", 0.0, 15.0, 4.0, 0.1) / 100
fee_rate = st.sidebar.slider("Tỷ lệ phí giao dịch (%)", 0.0, 1.0, 0.15, 0.01) / 100
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=365, value=252)

rf_daily = rf_annual / trading_days

# TÌM NGÀY OOS
available_dates = close.index
min_year = available_dates.min().year
max_year = available_dates.max().year

oos_start_date = st.sidebar.date_input(
    "Ngày bắt đầu đầu tư (Out-of-Sample)",
    value=datetime(2021, 1, 4),
    min_value=available_dates.min().to_pydatetime(),
    max_value=available_dates.max().to_pydatetime()
)

# THAM SỐ CHIẾN LƯỢC (Tùy chỉnh thủ công)
st.sidebar.header("🛠️ Tham Số Chiến Lược")
manual_bb_win = st.sidebar.slider("BB Window (Chu kỳ)", 10, 40, int(st.session_state.best_p["bb_window"]))
manual_bb_std = st.sidebar.slider("BB Standard Deviation (Độ lệch chuẩn)", 1.5, 3.0, float(st.session_state.best_p["bb_std"]), 0.1)
manual_rsi_win = st.sidebar.slider("RSI Window", 5, 30, int(st.session_state.best_p["rsi_window"]))
manual_vol_win = st.sidebar.slider("Volume Window", 10, 40, int(st.session_state.best_p["vol_window"]))
manual_lower = st.sidebar.slider("Ngưỡng RSI quá bán (Mua)", 15.0, 50.0, float(st.session_state.best_p["lower"]), 0.5)
manual_upper = st.sidebar.slider("Ngưỡng RSI quá mua (Bán)", 50.0, 90.0, float(st.session_state.best_p["upper"]), 0.5)

# Cập nhật tham số hiện tại dựa trên sidebar điều chỉnh thủ công
current_params = {
    "bb_window": manual_bb_win,
    "bb_std": manual_bb_std,
    "rsi_window": manual_rsi_win,
    "vol_window": manual_vol_win,
    "lower": manual_lower,
    "upper": manual_upper
}

# ----------------- TABS GIAO DIỆN CHÍNH -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Khám phá Dữ liệu", 
    "🧬 Tối ưu hóa (PSO)", 
    "📉 Kiểm thử Danh mục (Backtest)", 
    "🔍 Phân tích kỹ thuật chi tiết"
])

# ----------------- TAB 1: KHÁM PHÁ DỮ LIỆU -----------------
with tab1:
    st.subheader("🔍 Tổng quan bộ dữ liệu HOSE")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Tổng số cổ phiếu</div>
            <div class="metric-value">{len(STOCKS)}</div>
            <div class="metric-sub">Không tính chỉ số VNINDEX</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ngày bắt đầu</div>
            <div class="metric-value">{available_dates.min().strftime('%d/%m/%Y')}</div>
            <div class="metric-sub">Dữ liệu thô đầu kỳ</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ngày kết thúc</div>
            <div class="metric-value">{available_dates.max().strftime('%d/%m/%Y')}</div>
            <div class="metric-sub">Dữ liệu thô cuối kỳ</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Tổng số dòng dữ liệu</div>
            <div class="metric-value">{len(df_raw):,}</div>
            <div class="metric-sub">Bản ghi giao dịch</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("📈 Tra cứu giá & khối lượng cổ phiếu")
    selected_stock = st.selectbox("Chọn mã cổ phiếu cần phân tích dữ liệu lịch sử:", STOCKS)
    
    stock_df = df_raw[df_raw["ticker"] == selected_stock].sort_values("date")
    
    # Biểu đồ Plotly cho dữ liệu lịch sử
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=stock_df["date"], y=stock_df["adj_close"], name="Giá đóng cửa điều chỉnh", line=dict(color="#1f4068", width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=stock_df["date"], y=stock_df["volume"], name="Khối lượng giao dịch", marker_color="rgba(108, 117, 125, 0.4)"), secondary_y=True)
    
    fig.update_layout(
        title=f"Lịch sử Giá và Khối lượng giao dịch của {selected_stock}",
        xaxis_title="Thời gian",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)")
    )
    fig.update_yaxes(title_text="Giá (VND)", secondary_y=False)
    fig.update_yaxes(title_text="Khối lượng", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📄 Xem bảng dữ liệu 10 dòng gần nhất")
    st.dataframe(stock_df.tail(10).round(2), use_container_width=True)

# ----------------- TAB 2: TỐI ƯU HÓA THAM SỐ (PSO) -----------------
with tab2:
    st.subheader("🧬 Thuật toán tối ưu bầy đàn (PSO - Particle Swarm Optimization)")
    st.markdown("""
    Thuật toán PSO mô phỏng hành vi di chuyển kiếm ăn của một bầy chim hoặc bầy cá để tìm ra bộ tham số kỹ thuật tối ưu.
    * Giai đoạn **In-Sample** (Học máy) sẽ được cấu hình trước ngày bắt đầu đầu tư (OOS). Cụ thể là dữ liệu từ đầu kỳ đến trước ngày **{0}**.
    * Mục tiêu là tìm bộ tham số chiến lược (BB, RSI, Volume) sao cho **Sharpe Ratio** trung bình của tất cả cổ phiếu đạt giá trị lớn nhất.
    """.format(oos_start_date.strftime('%d/%m/%Y')))

    col_pso1, col_pso2 = st.columns(2)
    with col_pso1:
        n_particles = st.slider("Số lượng phần tử trong bầy (Particles)", 5, 50, 15, help="Số lượng phần tử càng nhiều, khả năng hội tụ càng tốt nhưng thời gian tính toán lâu hơn.")
    with col_pso2:
        n_iterations = st.slider("Số lượng vòng lặp tối đa (Iterations)", 5, 50, 15, help="Số lượng vòng lặp càng nhiều, thuật toán càng có cơ hội tìm ra kết quả tối ưu hơn.")
    
    pso_seed = st.number_input("Random Seed", value=42)

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    run_optimization = st.button("🚀 BẮT ĐẦU CHẠY PSO TỐI ƯU HÓA THAM SỐ", type="primary")

    if run_optimization:
        # Chuẩn bị dữ liệu In-Sample
        is_mask = (close.index < pd.Timestamp(oos_start_date))
        if is_mask.sum() == 0:
            st.error("Không tìm thấy dữ liệu In-Sample hợp lệ! Vui lòng chọn ngày bắt đầu đầu tư lùi lại để có thời gian học.")
        else:
            close_is = close.loc[is_mask, STOCKS]
            open_is  = open_.loc[is_mask, STOCKS]
            vol_is   = volume.loc[is_mask, STOCKS]

            st.info(f"Kích thước In-Sample: {is_mask.sum()} phiên giao dịch.")
            
            # Định nghĩa hàm fitness tính Sharpe trung bình
            def fitness_function(p):
                dp = bbrsi_vol_signals_matrix(close_is, vol_is, p)
                eq, _ = backtest_matrix(open_is, close_is, dp, capital=capital, fee=fee_rate)
                sh = sharpe_vector(eq, rf_daily=rf_daily, periods=trading_days)
                sh = sh[np.isfinite(sh)]
                return float(sh.mean()) if len(sh) > 0 else -9.0

            # Tiến hành tối ưu hóa
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_params, best_value = pso_optimize(
                fitness_function, 
                n_particles=n_particles, 
                n_iter=n_iterations, 
                seed=pso_seed,
                progress_bar=progress_bar,
                status_text=status_text
            )
            
            st.session_state.best_p = best_params
            st.success("Tối ưu hóa hoàn tất thành công!")
            
            # 2. Xếp hạng và chọn TOP cổ phiếu dựa trên Sharpe In-Sample
            dp_is = bbrsi_vol_signals_matrix(close_is, vol_is, best_params)
            eq_is, ntr_is = backtest_matrix(open_is, close_is, dp_is, capital=capital, fee=fee_rate)
            
            rows = []
            for j, tk in enumerate(STOCKS):
                m = perf_metrics(eq_is[:, j], int(ntr_is[j]), initial=capital, rf_daily=rf_daily, periods=trading_days)
                rows.append({
                    "ticker": tk, 
                    "IS_Return%": m["Total Return [%]"], 
                    "IS_Sharpe": m["Sharpe"],
                    "IS_Sortino": m["Sortino"], 
                    "IS_MaxDD%": m["Max Drawdown [%]"], 
                    "IS_Trades": int(ntr_is[j])
                })
            
            rank_df = pd.DataFrame(rows).sort_values("IS_Sharpe", ascending=False).reset_index(drop=True)
            st.session_state.rank_df = rank_df
            st.session_state.top5 = rank_df.head(5)["ticker"].tolist()
            st.session_state.is_ret = {tk: float(rank_df.loc[rank_df.ticker == tk, "IS_Return%"].iloc[0]) for tk in st.session_state.top5}
            
            # Hiển thị kết quả tối ưu
            st.markdown("### 🎯 Kết quả tìm kiếm tham số tối ưu")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Bộ tham số tối ưu tìm được:**")
                st.json(best_params)
            with col_res2:
                st.write("**Chỉ số hiệu quả:**")
                st.metric("Sharpe Ratio Trung bình (In-Sample)", f"{best_value:.4f}")
                st.write("Top 5 cổ phiếu tối ưu nhất:")
                st.write(", ".join(st.session_state.top5))

    # Nếu đã có tham số tối ưu trong session, hiển thị bảng xếp hạng
    if st.session_state.rank_df is not None:
        st.markdown("---")
        st.subheader("🏆 Xếp hạng cổ phiếu trong giai đoạn In-Sample (Học máy)")
        st.dataframe(st.session_state.rank_df.round(3), use_container_width=True)

# ----------------- TAB 3: BACKTEST DANH MỤC (OUT-OF-SAMPLE) -----------------
with tab3:
    st.subheader("📉 Chạy Backtest và So sánh Danh mục (Out-of-Sample)")
    st.markdown("""
    Giai đoạn **Out-of-Sample** (Đầu tư thực tế) bắt đầu từ ngày **{0}** đến ngày **{1}**.
    Mô phỏng sẽ so sánh hiệu suất giữa:
    1. **Chiến lược tối ưu hóa**: Rổ cổ phiếu Top K sử dụng các tín hiệu và tái cân bằng định kỳ.
    2. **Buy & Hold Rổ Top K**: Phân bổ đều và nắm giữ rổ cổ phiếu Top K từ ngày đầu Out-of-Sample.
    3. **VN-Index**: Nắm giữ chỉ số chung của thị trường.
    4. **Equal-weight toàn thị trường**: Đầu tư phân bổ đều vào tất cả các mã cổ phiếu trên sàn HOSE (Mô phỏng như một quỹ ETF thị trường).
    """.format(oos_start_date.strftime('%d/%m/%Y'), available_dates.max().strftime('%d/%m/%Y')))

    col_back1, col_back2, col_back3 = st.columns(3)
    
    with col_back1:
        # Số lượng cổ phiếu trong rổ
        k_stocks = st.slider("Số lượng cổ phiếu trong rổ (Top K Sharpe In-Sample)", 2, 10, 5)
    with col_back2:
        # Cơ chế phân bổ tỷ trọng
        weight_scheme = st.selectbox("Cơ chế phân bổ tỷ trọng", ["equal", "performance"], format_func=lambda x: "Đều nhau (Equal)" if x=="equal" else "Theo hiệu quả (Performance-weighted)")
    with col_back3:
        # Chu kỳ tái cân bằng
        rebal_freq = st.selectbox("Chu kỳ tái cân bằng danh mục (Rebalancing)", ["none", "monthly", "quarterly", "annual"], index=2, format_func=lambda x: {
            "none": "Không tái cân bằng (Bám giữ vị thế)",
            "monthly": "Tái cân bằng Hàng tháng",
            "quarterly": "Tái cân bằng Hàng quý (Mặc định)",
            "annual": "Tái cân bằng Hàng năm"
        }.get(x))

    # Cập nhật danh sách rổ cổ phiếu tương ứng với k_stocks
    if st.session_state.rank_df is not None:
        basket = st.session_state.rank_df.head(k_stocks)["ticker"].tolist()
        # Lấy In-Sample return để tính trọng số Performance ban đầu
        is_ret_dict = {tk: float(st.session_state.rank_df.loc[st.session_state.rank_df.ticker == tk, "IS_Return%"].iloc[0]) for tk in basket}
    else:
        # Nếu chưa chạy PSO, lấy top 5 mặc định
        # (Ở đây ta sẽ chạy nhanh một lượt in-sample để tìm xếp hạng mặc định nếu người dùng chưa bấm tối ưu)
        is_mask_def = (close.index < pd.Timestamp(oos_start_date))
        close_is_def = close.loc[is_mask_def, STOCKS]
        open_is_def  = open_.loc[is_mask_def, STOCKS]
        vol_is_def   = volume.loc[is_mask_def, STOCKS]
        
        dp_is_def = bbrsi_vol_signals_matrix(close_is_def, vol_is_def, st.session_state.best_p)
        eq_is_def, ntr_is_def = backtest_matrix(open_is_def, close_is_def, dp_is_def, capital=capital, fee=fee_rate)
        
        rows_def = []
        for j, tk in enumerate(STOCKS):
            m = perf_metrics(eq_is_def[:, j], int(ntr_is_def[j]), initial=capital, rf_daily=rf_daily, periods=trading_days)
            rows_def.append({"ticker": tk, "IS_Sharpe": m["Sharpe"], "IS_Return%": m["Total Return [%]"]})
        
        rank_df_def = pd.DataFrame(rows_def).sort_values("IS_Sharpe", ascending=False).reset_index(drop=True)
        basket = rank_df_def.head(k_stocks)["ticker"].tolist()
        is_ret_dict = {tk: float(rank_df_def.loc[rank_df_def.ticker == tk, "IS_Return%"].iloc[0]) for tk in basket}

    st.markdown('<div class="highlight-box"><strong>Rổ cổ phiếu được chọn: </strong>' + ', '.join(basket) + '</div>', unsafe_allow_html=True)
    
    run_backtest = st.button("📊 CHẠY MÔ PHỎNG BACKTEST DANH MỤC", type="primary")

    if run_backtest:
        # Chạy portfolio chiến lược
        eq_strat, m_strat = run_portfolio(
            close, open_, volume, basket, current_params, weight_scheme, rebal_freq, is_ret_dict,
            capital=capital, fee=fee_rate, oos_start_date=oos_start_date, rf_daily=rf_daily, periods=trading_days
        )
        
        # Chạy benchmark Buy & Hold rổ
        eq_bh, m_bh = buy_hold_basket(
            close, open_, basket, capital=capital, fee=fee_rate, oos_start_date=oos_start_date, rf_daily=rf_daily, periods=trading_days
        )
        
        # Chạy benchmark VN-Index
        eq_idx, m_idx = buy_hold_index(
            close, "VNINDEX", capital=capital, oos_start_date=oos_start_date, rf_daily=rf_daily, periods=trading_days
        )
        
        # Chạy benchmark Equal-weight toàn thị trường
        eq_eqw, m_eqw = buy_hold_basket(
            close, open_, STOCKS, capital=capital, fee=fee_rate, oos_start_date=oos_start_date, rf_daily=rf_daily, periods=trading_days
        )

        # 1. Vẽ đồ thị tăng trưởng vốn (Equity Curve)
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat / capital, name=f"Chiến lược BB+RSI+Vol (Top {k_stocks})", line=dict(color="#1f4068", width=3)))
        fig_eq.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh / capital, name="Buy & Hold rổ Top K", line=dict(color="#43d8c4", width=2, dash='dash')))
        fig_eq.add_trace(go.Scatter(x=eq_eqw.index, y=eq_eqw / capital, name="Equal-weight toàn 99 mã (ETF Proxy)", line=dict(color="#9b59b6", width=2, dash='dot')))
        fig_eq.add_trace(go.Scatter(x=eq_idx.index, y=eq_idx / capital, name="VN-Index", line=dict(color="#e43f5a", width=2.5)))
        
        fig_eq.update_layout(
            title="So sánh Tăng trưởng Vốn đầu tư (Out-of-Sample Period)",
            xaxis_title="Thời gian",
            yaxis_title="Tỷ lệ tăng trưởng (x lần vốn ban đầu)",
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.5)"),
            height=500
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # 2. Hiển thị Bảng so sánh chỉ số hiệu năng
        st.subheader("📋 Bảng so sánh các chỉ số hiệu quả đầu tư")
        
        comparison_data = {
            "Chỉ số hiệu năng": [
                "Tổng Lợi Nhuận [%]", "CAGR [%] (Lợi nhuận gộp năm)", "Độ Biến Động [%]", 
                "Sharpe Ratio", "Sortino Ratio", "Mức Sụt Giảm Lớn Nhất [%]", 
                "Calmar Ratio", "Số Lệnh Giao Dịch", "Tổng Phí Đã Trả (VND)", "Giá Trị Cuối Kỳ (VND)"
            ],
            f"Chiến lược (Top {k_stocks})": [
                m_strat["Total Return [%]"], m_strat["CAGR [%]"], m_strat["Volatility [%]"],
                m_strat["Sharpe"], m_strat["Sortino"], m_strat["Max Drawdown [%]"],
                m_strat["Calmar"], m_strat["Trades"], m_strat["Fees Paid"], m_strat["Final Value"]
            ],
            "Buy & Hold rổ Top K": [
                m_bh["Total Return [%]"], m_bh["CAGR [%]"], m_bh["Volatility [%]"],
                m_bh["Sharpe"], m_bh["Sortino"], m_bh["Max Drawdown [%]"],
                m_bh["Calmar"], m_bh["Trades"], 0.0, m_bh["Final Value"]
            ],
            "Equal-weight toàn bộ mã": [
                m_eqw["Total Return [%]"], m_eqw["CAGR [%]"], m_eqw["Volatility [%]"],
                m_eqw["Sharpe"], m_eqw["Sortino"], m_eqw["Max Drawdown [%]"],
                m_eqw["Calmar"], m_eqw["Trades"], 0.0, m_eqw["Final Value"]
            ],
            "VN-Index": [
                m_idx["Total Return [%]"], m_idx["CAGR [%]"], m_idx["Volatility [%]"],
                m_idx["Sharpe"], m_idx["Sortino"], m_idx["Max Drawdown [%]"],
                m_idx["Calmar"], m_idx["Trades"], 0.0, m_idx["Final Value"]
            ]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Định dạng hiển thị
        formatted_df = comp_df.copy()
        for col in formatted_df.columns[1:]:
            # Dòng 0, 1, 2, 5 hiển thị %
            formatted_df.iloc[0, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[0, col]:.2f}%"
            formatted_df.iloc[1, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[1, col]:.2f}%"
            formatted_df.iloc[2, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[2, col]:.2f}%"
            formatted_df.iloc[5, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[5, col]:.2f}%"
            # Dòng 3, 4, 6 hiển thị số thực 2 chữ số thập phân
            formatted_df.iloc[3, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[3, col]:.2f}"
            formatted_df.iloc[4, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[4, col]:.2f}"
            formatted_df.iloc[6, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[6, col]:.2f}"
            # Dòng 7 là số lệnh giao dịch
            formatted_df.iloc[7, formatted_df.columns.get_loc(col)] = f"{int(comp_df.loc[7, col])}"
            # Dòng 8, 9 hiển thị định dạng tiền VND
            formatted_df.iloc[8, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[8, col]:,.0f} đ"
            formatted_df.iloc[9, formatted_df.columns.get_loc(col)] = f"{comp_df.loc[9, col]:,.0f} đ"

        st.dataframe(formatted_df, use_container_width=True)

        # 3. Phân tích chi tiết theo từng năm
        st.subheader("📅 Hiệu suất chi tiết qua các năm")
        S = submetrics(eq_strat, "Chiến lược", capital, rf_daily, trading_days)
        B = submetrics(eq_bh, "Buy & Hold rổ", capital, rf_daily, trading_days)
        V = submetrics(eq_idx, "VN-Index", capital, rf_daily, trading_days)
        
        years_list = sorted(list(set(eq_strat.index.year)))
        
        yr_rows = []
        for yr in years_list:
            yr_str = str(yr)
            yr_rows.append({
                "Năm": yr_str,
                "Lợi nhuận Chiến lược [%]": f"{S[yr_str]['Total Return [%]']:.2f}%",
                "MaxDD Chiến lược [%]": f"{S[yr_str]['Max Drawdown [%]']:.2f}%",
                "Lợi nhuận Buy&Hold [%]": f"{B[yr_str]['Total Return [%]']:.2f}%",
                "MaxDD Buy&Hold [%]": f"{B[yr_str]['Max Drawdown [%]']:.2f}%",
                "Lợi nhuận VN-Index [%]": f"{V[yr_str]['Total Return [%]']:.2f}%",
                "MaxDD VN-Index [%]": f"{V[yr_str]['Max Drawdown [%]']:.2f}%",
            })
        st.dataframe(pd.DataFrame(yr_rows), use_container_width=True)

        # 4. Kiểm định thống kê
        if HAVE_SCIPY:
            st.subheader("🧪 Kiểm định thống kê (Statistical Hypothesis Testing)")
            r_s = eq_strat.values[1:] / eq_strat.values[:-1] - 1
            r_b = eq_bh.values[1:] / eq_bh.values[:-1] - 1
            r_i = eq_idx.values[1:] / eq_idx.values[:-1] - 1
            
            # Một số mẫu có thể bị trùng hoặc không đủ chiều dài
            if len(r_s) > 10:
                t1 = ttest_1samp(r_s - rf_daily, 0, alternative="greater")
                wb = wilcoxon(r_s, r_b, alternative="greater")
                wi = wilcoxon(r_s, r_i, alternative="greater")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("t-test (Lợi nhuận > Lãi suất phi rủi ro)", f"p-value: {t1.pvalue:.2e}")
                    st.markdown("👉 **Ý nghĩa:** " + ("CÓ ý nghĩa thống kê (p < 0.05)" if t1.pvalue < 0.05 else "Không đủ ý nghĩa thống kê (p >= 0.05)"))
                with col_stat2:
                    st.metric("Wilcoxon vs Buy&Hold rổ Top K", f"p-value: {wb.pvalue:.2e}")
                    st.markdown("👉 **Ý nghĩa:** " + ("CÓ ý nghĩa thống kê (p < 0.05)" if wb.pvalue < 0.05 else "Không đủ ý nghĩa thống kê (p >= 0.05)"))
                with col_stat3:
                    st.metric("Wilcoxon vs VN-Index", f"p-value: {wi.pvalue:.2e}")
                    st.markdown("👉 **Ý nghĩa:** " + ("CÓ ý nghĩa thống kê (p < 0.05)" if wi.pvalue < 0.05 else "Không đủ ý nghĩa thống kê (p >= 0.05)"))
            else:
                st.warning("Không đủ dữ liệu giao dịch để thực hiện kiểm định thống kê tin cậy.")

# ----------------- TAB 4: BIỂU ĐỒ KỸ THUẬT CHI TIẾT -----------------
with tab4:
    st.subheader("🔍 Phân tích tín hiệu giao dịch chi tiết")
    st.markdown("""
    Chọn một cổ phiếu để xem chi tiết các đường **Bollinger Bands**, chỉ số **RSI**, **Khối lượng giao dịch**, 
    và các thời điểm chiến lược đưa ra tín hiệu **Mua (Mũi tên xanh chỉ lên)** và **Bán (Mũi tên đỏ chỉ xuống)**.
    """)
    
    # Chỉ cho chọn các cổ phiếu trong rổ hoặc top stocks để vẽ cho nhanh
    selected_chart_stock = st.selectbox("Chọn mã cổ phiếu cần vẽ biểu đồ kỹ thuật:", basket)
    
    # Lấy dữ liệu Out-of-Sample của mã được chọn
    oos_start_ts = pd.Timestamp(oos_start_date)
    close_oos = close.loc[close.index >= oos_start_ts, selected_chart_stock]
    vol_oos = volume.loc[volume.index >= oos_start_ts, selected_chart_stock]
    
    if close_oos.dropna().empty or vol_oos.dropna().empty:
        st.warning("Không đủ dữ liệu Out-of-sample của cổ phiếu này để hiển thị.")
    else:
        # Tính toán lại các thông số kỹ thuật cho cổ phiếu này dựa trên bộ tham số hiện tại
        sma = close_oos.rolling(window=int(current_params["bb_window"]), min_periods=1).mean()
        std = close_oos.rolling(window=int(current_params["bb_window"]), min_periods=1).std()
        upper_band = sma + current_params["bb_std"] * std
        lower_band = sma - current_params["bb_std"] * std
        
        # RSI
        delta = close_oos.diff()
        ag = delta.clip(lower=0).rolling(window=int(current_params["rsi_window"]), min_periods=1).mean()
        al = -delta.clip(upper=0).rolling(window=int(current_params["rsi_window"]), min_periods=1).mean()
        rs = ag / al.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)
        
        # Vol SMA
        vol_sma = vol_oos.rolling(window=int(current_params["vol_window"]), min_periods=1).mean()
        
        # Tín hiệu Mua/Bán
        buy_signals = (close_oos <= lower_band) & (rsi < current_params["lower"]) & (vol_oos > vol_sma)
        sell_signals = (close_oos >= upper_band) & (rsi > current_params["upper"])
        
        buy_prices = close_oos[buy_signals]
        sell_prices = close_oos[sell_signals]
        
        # Vẽ đồ thị 3 subplot bằng Plotly
        fig_tech = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Đường giá + BB + Tín hiệu Mua/Bán
        fig_tech.add_trace(go.Scatter(x=close_oos.index, y=close_oos, name="Giá đóng cửa điều chỉnh", line=dict(color="black", width=1.5)), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=sma.index, y=sma, name="Bollinger Band SMA", line=dict(color="gray", width=1, dash='dash')), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=upper_band.index, y=upper_band, name="Upper Band", line=dict(color="green", width=1, dash='dot')), row=1, col=1)
        fig_tech.add_trace(go.Scatter(x=lower_band.index, y=lower_band, name="Lower Band", line=dict(color="red", width=1, dash='dot')), row=1, col=1)
        
        # Thêm vùng Bollinger Bands tô màu
        fig_tech.add_trace(go.Scatter(
            x=list(close_oos.index) + list(close_oos.index)[::-1],
            y=list(upper_band) + list(lower_band)[::-1],
            fill='toself',
            fillcolor='rgba(173, 181, 189, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ), row=1, col=1)
        
        # Đánh dấu tín hiệu Mua/Bán
        fig_tech.add_trace(go.Scatter(
            x=buy_prices.index, y=buy_prices, 
            mode='markers', 
            marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='black')), 
            name='Tín hiệu MUA'
        ), row=1, col=1)
        
        fig_tech.add_trace(go.Scatter(
            x=sell_prices.index, y=sell_prices, 
            mode='markers', 
            marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='black')), 
            name='Tín hiệu BÁN'
        ), row=1, col=1)
        
        # 2. Subplot RSI
        fig_tech.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="purple", width=1.5)), row=2, col=1)
        fig_tech.add_hline(y=current_params["upper"], line_dash="dash", line_color="red", row=2, col=1)
        fig_tech.add_hline(y=current_params["lower"], line_dash="dash", line_color="green", row=2, col=1)
        fig_tech.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
        
        # 3. Subplot Volume
        fig_tech.add_trace(go.Bar(x=vol_oos.index, y=vol_oos, name="Volume", marker_color="rgba(31, 64, 104, 0.7)"), row=3, col=1)
        fig_tech.add_trace(go.Scatter(x=vol_sma.index, y=vol_sma, name="Volume SMA", line=dict(color="orange", width=1.5)), row=3, col=1)
        fig_tech.update_yaxes(title_text="Volume", row=3, col=1)
        
        fig_tech.update_layout(
            title=f"Phân tích kỹ thuật chi tiết của cổ phiếu {selected_chart_stock} (Out-of-Sample)",
            xaxis3_title="Thời gian",
            hovermode="x unified",
            height=700,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tech, use_container_width=True)
