import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from scipy.spatial.distance import cdist
import numpy as np
import random
import datetime
import csv
import os
import __Custom_Functions_Backtest as cf_bt


# 分數矩陣運算
def calculate_score(metrics, weights, ranges):
    """
    根據指標和權重計算加權總分，並對指標進行區間標準化
    Args:
        metrics (dict): 包含多個指標的字典
        weights (dict): 每個指標的權重
        ranges (dict): 每個指標的標準化區間 {'指標名稱': (min_val, max_val)}
    Returns:
        float: 加權總分
    """

    score = 0
    for key, weight in weights.items():
        if key in metrics:
            value = metrics[key]
            if key in ranges:  # 如果有定義標準化區間，進行標準化
                min_val, max_val = ranges[key]
                value = normalize(value, min_val, max_val)
            score += value * weight
    return score

# 標準化
def normalize(value, min_val, max_val):
    """
    將值標準化到 [0, 1] 區間
    Args:
        value (float): 原始值
        min_val (float): 指標的最小值
        max_val (float): 指標的最大值
    Returns:
        float: 標準化後的值
    """
    if value > max_val:  # 限制在範圍內
        value = max_val
    if value < min_val:
        value = min_val
    return (value - min_val) / (max_val - min_val)

# 迭代器
def optimize_multiple_times(n_iterations, n_calls, diversity_threshold, acq_func):
    global found_params
    results = []

    # 欄位順序根據 weights
    field_order = list(weights.keys()) + [
        'iteration', 'best_score', 'seed', 'best_params', 'py_filename', 'time'
    ]
    seen = set()
    field_order = [x for x in field_order if not (x in seen or seen.add(x))]

    # 欄位對應中文（自訂）
    field_map_zh = {
        'gross_profit': "總收益率",
        'sharpe_ann': "年化Sharpe",
        'calmar_ann': "年化Calmar",
        'profit_factor': "獲利因子",
        'max_drawdown': "最大連續虧損",
        'trades': "交易次數",
        'win_rate': "勝率",
        'positive_month_proportion': "正收益月比例",
        'avg_loss_negative_months': "負收益月平均虧損",
        'iteration': "迭代",
        'best_score': "最佳分數",
        'seed': "隨機種子",
        'best_params': "最佳參數",
        'py_filename': "程式名",
        'time': "時間"
    }

    for i in range(n_iterations):
        seed = random.randint(0, int(1e6))
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=seed, acq_func=acq_func)
        found_params.append(result.x)
        results.append((-result.fun, result.x))

        print(f"🔥🔥🔥 結束第 {i + 1} 次迭代 🔥🔥🔥")
        print(f"{datetime.datetime.now()}")
        print(f"本次迭代最佳分數: {-result.fun:.4f}🔥 隨機種子(Seed): {seed}")
        print(f"本次迭代最佳參數: {[float(x) for x in result.x]}")

        temp_df = target_df.copy()
        temp_df = cf_bt.generate_signals_ai_20250616_164528_4(temp_df, *result.x)  ###################
        trades = cf_bt.backtest(temp_df, slippage=slippage)
        trades_df = cf_bt.process_trades_to_df(trades)
        results_temp = cf_bt.calculate_stats(trades_df)

        # ====== 根據 weights 順序統一 print ======
        print("｜".join([
            f"{field_map_zh.get(k,k)}: {results_temp.get(k, '-'):.4f}" if isinstance(results_temp.get(k), (float, int))
            else f"{field_map_zh.get(k,k)}: {results_temp.get(k, '-')}"
            for k in weights.keys()
        ]))
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥\n")

        # 動態生成 row
        row = []
        for key in field_order:
            if key == 'time':
                row.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-4])
            elif key == 'iteration':
                row.append(i + 1)
            elif key == 'best_score':
                row.append(round(-result.fun, 2))
            elif key == 'seed':
                row.append(seed)
            elif key == 'best_params':
                row.append([float(x) for x in result.x])
            elif key == 'py_filename':
                row.append(os.path.basename(__file__))
            else:
                row.append(results_temp.get(key, None))

        current_python_file = os.path.splitext(os.path.basename(__file__))[0]
        csv_file = f'optimization_results_{current_python_file}.csv'
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(field_order)
            writer.writerow(row)
        print(os.path.basename(__file__))

    # 篩選唯一結果
    unique_results = []
    for profit, params in results:
        if all(
            np.linalg.norm(np.array(params) - np.array(uniq_params)) > diversity_threshold
            for _, uniq_params in unique_results
        ):
            unique_results.append((profit, params))

    unique_results = sorted(unique_results, key=lambda x: -x[0])
    return unique_results


# 目標回測函式
def objective(params):
    try:
        temp_df = target_df.copy()
        temp_df = cf_bt.generate_signals_ai_20250616_164528_4(temp_df, *params) ###################
        trades = cf_bt.backtest(temp_df, slippage=slippage)
        if not trades:  # 檢查是否為無效回測
            print(f"Invalid backtest. Params: {[float(x) for x in params]}")
            return 1e8
        trades_df = cf_bt.process_trades_to_df(trades)
        if trades_df.empty:
            print(f"Invalid backtest. Params: {[float(x) for x in params]}")
            return 1e8
        results = cf_bt.calculate_stats(trades_df)

        # 交易次數檢查
        trades = results.get('trades', 0)
        Gross_Profit = results.get('gross_profit', 0)

        # ========== 開始計算分數 ==========
        metrics = {key: results.get(key, 0) for key in weights.keys()}

        score = calculate_score(metrics, weights, ranges)
        if trades < Trades_MIN :
            print(f"Penalty__ trades too low: {trades}. Penalty. Params: {[float(x) for x in params]}")
            return 1e8
        if trades > Trades_MAX :
            print(f"Penalty__ trades too high: {trades}. Penalty. Params: {[float(x) for x in params]}")
            return 1e8
        if Gross_Profit < 0 :
            print(f"Penalty__ Gross_Profit too low: {Gross_Profit} Params: {[float(x) for x in params]}")
            return 1e8 
        if not np.isfinite(score):
            print(f"Invalid score: {score}, Params: {params}")
            return 1e8
    except Exception as e:
        print(f"Error during backtest with params {params}: {e}")
        return 1e8

    # 如果有多樣性懲罰可放這裡 (保持原樣)
    if found_params:
        normalized_params = np.array([[(p - s.low) / (s.high - s.low) for p, s in zip(params, space)]])
        normalized_found_params = np.array([[(fp - s.low) / (s.high - s.low) for fp, s in zip(fp_set, space)] for fp_set in found_params])
        distances = cdist(normalized_params, normalized_found_params, metric='euclidean')
        diversity_penalty = max(np.min(distances), 1e-8)
    else:
        diversity_penalty = 1
    return -score + 0.1 / diversity_penalty

# # 定義滑價
slippage = 0.0002

# 定義最大交易 最小交易 次數
Trades_MIN = 60
Trades_MAX = 1200 # 平均每月交易60筆 
benchmark_gross_profit = 1.1 # 目標標的1口小台在目標區間的收益率為 100 > 210 110% Plus 
# 定義迭代次數 每次迭代的batch 標籤多樣性 越高越多樣
# 'gp_hedge'（默認）：平衡探索 | 'LCB'：探索未測試區域 | 'EI' 或 'PI'：利用已知高分數區域
Total_Iterations = 10000
Total_Calls = 20
Diversity_Threshold = 0.0001
# Acq_Func = 'gp_hedge'
Acq_Func = 'LCB'

# 定義搜索空間 ###################
space = [
    Integer(100, 40000, name='bb_window'),
    Real(0, 20, name='bb_std_dev_multiplier'),
    Integer(100, 40000, name='mfi_window'),
    Integer(100, 40000, name='atr_window'),
    Integer(100, 40000, name='atr_norm_window'),
    Real(0, 20, name='vol_stability_threshold'),
]

# 定義權重
weights = {
    'gross_profit': 0.5,
    'sharpe_ann': 0.5,
    'calmar_ann': 0.5,
    'profit_factor': 0.5,
    'max_drawdown': -0.5,
    'trades': 0.1,
    'win_rate': 0.01,
    'positive_month_proportion': 0.01,
    'avg_loss_negative_months': -0.01,
}

# 定義指標範圍
ranges = {
    'gross_profit': (0, benchmark_gross_profit),
    'sharpe_ann': (-1, 3),
    'calmar_ann': (-1, 5),
    'profit_factor': (0, 3),
    'max_drawdown': (-0.99, 0),
    'trades': (Trades_MIN, Trades_MAX),
    'win_rate': (0.01, 1),
    'positive_month_proportion': (0.01, 1),
    'avg_loss_negative_months': (-0.2, 0),
}

# 定義回測範圍
target_df = pd.read_feather('RESULT_fut_gap_fixed_2024.feather')
# 執行優化
found_params, all_results = [], []
high_profit_params = optimize_multiple_times(n_iterations=Total_Iterations, n_calls=Total_Calls, diversity_threshold=Diversity_Threshold, acq_func=Acq_Func)
print(f"💧💧💧 結束迭代 💧💧💧💧💧💧💧💧💧💧💧💧")





'''

請為這個函式定義正確的搜索範圍

def generate_signals_regime_filter(
    df: pd.DataFrame,
    # ── regime filter ──────────────────────────────
    tc_window: int = 300,             # true_cap 斜率的回看窗
    slope_thresh: float = 0.0005,     # true_cap 斜率門檻
    # ── trend breakout ────────────────────────────
    bb_window: int = 50,              # Bollinger 頻寬
    bb_std_mult: float = 2.0,         # Bollinger 標準差倍數
    macd_fast: int = 12, macd_slow: int = 26, macd_sig: int = 9,
    # ── range reversal ────────────────────────────
    wr_period: int = 14,
    wr_long_thresh: float = 20,       # oversold
    wr_short_thresh: float = 80,      # overbought
    mfi_period: int = 14,
    mfi_long_thresh: float = 20,
    mfi_short_thresh: float = 80
):

使用這個格式
space = [
    # ── Event detection ────────────────────────────────
    Real    (0.005, 0.05,   name="gap_pct"),            # 0.5%–5% opening jump
    Real    (1.0,   5.0,    name="vol_spike_mult"),     # 1×–5× average volume
    Integer (30,    12000,    name="vol_window"),         # 30–300 bars for volume rolling mean

    # ── Confirmation (Williams %R) ─────────────────────
    Integer (5,     12000,     name="wr_window"),          # 5–60 bars lookback
    Integer (1,     30,     name="wr_entry_thresh"),    # 1%–30% oversold
    Integer (70,    99,     name="wr_exit_thresh"),     # 70%–99% overbought

    # ── Confirmation (Trend filter) ────────────────────
    Integer (10,    12000,    name="ma_window"),          # 10–200 bar moving average
]
'''

