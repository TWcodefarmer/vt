import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from scipy.spatial.distance import cdist
import numpy as np
import random
import datetime
from hashlib import md5

# indicator 
def indicator_macd_histogram(price_series, fast_window=5700, slow_window=3400, signal_window=1500):
    """
    # 會有大量不同的值 +600~-300 之類的 取決於window
    df['hist'] = indicator_macd_histogram(
            df['price'], 
            fast_window=macd_fast_window, 
            slow_window=macd_slow_window, 
            signal_window=macd_signal_window
        ).fillna(0)
    計算 MACD Histogram
    模擬 macd_indicator.hist 的功能
    Args:
        df: 包含價格資料的 DataFrame
        price_column: 用於計算 MACD 的價格欄位名稱
        fast_window: 快速 EMA 的窗口期
        slow_window: 慢速 EMA 的窗口期
        signal_window: 信號線 EMA 的窗口期
    Returns:
        hist: MACD Histogram 的 Series
    """
    # 計算快速和慢速 EMA
    fast_ema = price_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow_window, adjust=False).mean()
    
    # 計算 MACD 線和信號線
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    
    # 計算 MACD Histogram
    hist_series = macd_line - signal_line
    
    return hist_series

def indicator_williams_r(price_series, period):
    """
    # 產出0~100的值
    計算威廉指標 (Williams %R)
    """
    highest_high = price_series.rolling(window=period, min_periods=period).max()
    lowest_low = price_series.rolling(window=period, min_periods=period).min()
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)
    wr_series = ((highest_high - price_series) / (highest_high - lowest_low) * 100)
    return wr_series

def indicator_mfi(cap_series, price_series, period):
    """
    # 產出0~100的值
    計算 MFI 指標 (Money Flow Index)。

    Parameters:
        df (pd.DataFrame): 包含價格和成交量數據的 DataFrame。

    Returns:
        pd.DataFrame: 包含原始數據和計算出的 MFI 指標。
    """
    # 計算正負資金流量
    positive_flow_series = cap_series.where(price_series.diff() > 0, 0)
    negative_flow_series = cap_series.where(price_series.diff() < 0, 0)
    
    # 滾動窗口計算正負資金流總和
    sum_positive_flow_series = positive_flow_series.rolling(window=period, min_periods=period).sum()
    sum_negative_flow_series = negative_flow_series.rolling(window=period, min_periods=period).sum()
    
    # 計算資金流量比率和 MFI
    money_flow_ratio_series = sum_positive_flow_series / sum_negative_flow_series.replace(0, 1e-8)  # 避免除以零
    mfi_series = 100 - (100 / (1 + money_flow_ratio_series))
    return mfi_series

def indicator_bollinger_bands(price_series, window, std_dev_multiplier):
    """
    # 均線 不同period 產出價格相關的數字
    計算布林通道的上下軌和中軌。
    Args:
        price_series (pd.Series): 價格數據序列
        window (int): 移動平均的窗口大小
        std_dev_multiplier (float): 標準差的倍數，用於計算上下軌
    Returns:
        pd.DataFrame: upper_band_series, middle_band_series, lower_band_series
    """
    middle_band_series = price_series.rolling(window=window, min_periods=1).mean()
    std_dev = price_series.rolling(window=window, min_periods=1).std()
    upper_band_series = middle_band_series + std_dev_multiplier * std_dev
    lower_band_series = middle_band_series - std_dev_multiplier * std_dev
    return upper_band_series, middle_band_series, lower_band_series

def indicator_rsi(price_series, period):
    '''
    # 產出0~100的值
    '''
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-8)  # 避免除零
    rsi = 100 - (100 / (1 + rs))
    return rsi

def indicator_atr(price_series, period):
    '''
    # 產出0~10的值 適合做通道
    '''
    price_change = price_series.diff().abs()  # 每分鐘價格變動的絕對值
    atr = price_change.rolling(window=period, min_periods=period).mean()
    return atr

def indicator_volatility_annualized(price_series, period, freq=252):
    '''
    # 產出0~0.025的值
    '''
    returns = np.log(price_series / price_series.shift(1))
    vol = returns.rolling(window=period, min_periods=period).std()
    return vol * np.sqrt(freq)

def indicator_stochastic_kdj(price_series, period):
    '''
    # 會有大量不同的值 +600~-300 之類的 取決於window
    '''
    lowest_low = price_series.rolling(window=period, min_periods=period).min()
    highest_high = price_series.rolling(window=period, min_periods=period).max()
    rsv = 100 * (price_series - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = rsv.ewm(alpha=1/3).mean()
    d = k.ewm(alpha=1/3).mean()
    j = 3 * k - 2 * d
    return k, d, j

def indicator_cci(price_series, period):
    """
    # 會有大量不同的值 +600~-300 之類的 取決於window
    Compute Commodity Channel Index (CCI) in a vectorized way.

    CCI = (price - SMA(price, period)) / (0.015 * mean_abs_dev)

    where mean_abs_dev = rolling_mean( |price - SMA(price, period)| )
    """
    # Simple moving average
    ma = price_series.rolling(window=period, min_periods=period).mean()
    # Mean absolute deviation from that moving average
    dev = (price_series - ma).abs()
    md = dev.rolling(window=period, min_periods=period).mean()
    # CCI formula
    cci = (price_series - ma) / (0.015 * md)
    return cci

# indicator norm
def indicator_atr_norm(price_series, period, window):
    '''
    # 產出-1~1的值
    '''
    price_change = price_series.diff().abs()
    atr = price_change.ewm(span=period, min_periods=period, adjust=False).mean()
    atr_mean = atr.rolling(window, min_periods=window).mean()
    atr_std = atr.rolling(window, min_periods=window).std()
    atr_z = (atr - atr_mean) / (atr_std + 1e-8)
    atr_norm = pd.Series(np.tanh(atr_z))
    return atr_norm

def indicator_cci_norm(price_series, period, window):
    """
    # 產出-1~1的值
    計算 Commodity Channel Index (CCI) 並將其標準化到 [-1, 1]。
    先做 z-score 標準化，再用 tanh 收斂到 (-1, 1)。
    """
    # 原始CCI計算
    ma = price_series.rolling(window=period, min_periods=period).mean()
    dev = (price_series - ma).abs()
    md = dev.rolling(window=period, min_periods=period).mean()
    cci = (price_series - ma) / (0.015 * md)
    
    # z-score + tanh 標準化
    cci_mean = cci.rolling(window, min_periods=window).mean()
    cci_std = cci.rolling(window, min_periods=window).std()
    cci_z = (cci - cci_mean) / (cci_std + 1e-8)
    cci_norm = pd.Series(np.tanh(cci_z))
    return cci_norm

def indicator_kdj_norm(price_series, period, norm_window):
    '''
    # 產出-1~1的值
    '''
    # 計算原始KDJ
    lowest_low = price_series.rolling(window=period, min_periods=period).min()
    highest_high = price_series.rolling(window=period, min_periods=period).max()
    rsv = 100 * (price_series - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = rsv.ewm(alpha=1/3).mean()
    d = k.ewm(alpha=1/3).mean()
    j = 3 * k - 2 * d

    # z-score + tanh 標準化到 (-1, 1)
    k_z = (k - k.rolling(norm_window).mean()) / (k.rolling(norm_window).std() + 1e-8)
    d_z = (d - d.rolling(norm_window).mean()) / (d.rolling(norm_window).std() + 1e-8)
    j_z = (j - j.rolling(norm_window).mean()) / (j.rolling(norm_window).std() + 1e-8)

    k_norm = pd.Series(np.tanh(k_z))
    d_norm = pd.Series(np.tanh(d_z))
    j_norm = pd.Series(np.tanh(j_z))

    # 直接打包 DataFrame 給外部用
    return k_norm, d_norm, j_norm

def indicator_obv_norm(price_series, volume_series, window):
    '''
    # 產出-1~1的值
    '''
    direction = np.sign(price_series.diff())
    direction.iloc[0] = 0
    obv = (volume_series * direction).cumsum()
    obv_min = obv.rolling(window, min_periods=window).min()
    obv_max = obv.rolling(window, min_periods=window).max()
    obv_norm = (obv - obv_min) / (obv_max - obv_min + 1e-8)
    obv_norm = pd.Series(obv_norm * 2 - 1)
    return obv_norm

def indicator_macd_histogram_norm(price_series, fast_window=5700, slow_window=3400, signal_window=1500, norm_window=3000):
    """
    # 產出-1~1的值
    計算 MACD Histogram 並使用滾動窗口進行 Min-Max 標準化，使其值落在 -1 到 1。

    Args:
        price_series (pd.Series): 包含價格資料的 Series。
        fast_window (int): 快速 EMA 的窗口期。
        slow_window (int): 慢速 EMA 的窗口期。
        signal_window (int): 信號線 EMA 的窗口期。
        norm_window (int): 用於標準化的滾動窗口期。

    Returns:
        pd.Series: 標準化後的 MACD Histogram Series，值介於 -1 到 1。
    """
    # 步驟 1: 計算 MACD Histogram (與你原來的函式相同)
    fast_ema = price_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow_window, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    hist_series = macd_line - signal_line

    # 步驟 2: 使用滾動窗口計算最大值和最小值
    # min_periods=1 確保即使數據點不足一個完整窗口也能計算 (例如在時間序列的開頭)
    min_val_rolling = hist_series.rolling(norm_window, min_periods=1).min()
    max_val_rolling = hist_series.rolling(norm_window, min_periods=1).max()
    
    # 步驟 3: 執行 Min-Max Scaling
    # (hist_series - min_val) / (max_val - min_val) 會將值縮放到 0 到 1
    # 加上 1e-8 是為了防止分母為零，避免除以零的錯誤
    range_val_rolling = max_val_rolling - min_val_rolling
    
    # 處理分母為零的情況：如果窗口內的最大值和最小值相同，則該點的標準化值為 0
    # (這裡我們將其設為 0，或可以選擇 NaN)
    # np.where(condition, x, y) 如果 condition 為 True，取 x；否則取 y
    hist_norm_0_1 = np.where(
        range_val_rolling == 0,
        0, # 如果 range 為 0，則值為 0
        (hist_series - min_val_rolling) / (range_val_rolling)
    )
    
    # 步驟 4: 將 0 到 1 的範圍轉換為 -1 到 1
    hist_norm_neg1_1 = pd.Series(hist_norm_0_1 * 2 - 1)
    return hist_norm_neg1_1

# 回測模組
def backtest(df, slippage=0.0002):
    '''
    回測模組自動扣掉滑價  並且統計獲利並以實際日期排序
    假設同一根K棒出現平多與進空 那會在同一根K棒中先平多再進空
    ''' 
    position = 0  # 當前持倉狀態：1 = 多單, -1 = 空單, 0 = 無持倉
    entry_price = 0  # 進場價格
    trades = []  # 用來記錄交易細節

    # 提取交易信號的行索引
    signals = df[(df['long_entry'] == 1) | (df['long_exit'] == 1) |
                 (df['short_entry'] == 1) | (df['short_exit'] == 1)].reset_index()

    # 將數據轉為 NumPy 陣列以加速處理
    prices = signals['price'].to_numpy()
    long_entry = signals['long_entry'].to_numpy()
    long_exit = signals['long_exit'].to_numpy()
    short_entry = signals['short_entry'].to_numpy()
    short_exit = signals['short_exit'].to_numpy()
    indices = signals['index'].to_numpy()
    dt = signals['datetime'].to_numpy()

    for i in range(len(prices)):
        price = prices[i]
        slippage_cost = price * slippage

        # 多單邏輯
        if position == 0 and long_entry[i] == 1:  # 進多單
            position = 1
            entry_price = price + slippage_cost  # 進場價格加上滑價
            trades.append((indices[i], dt[i], 'Long_Entry', entry_price))
        elif position == 1 and long_exit[i] == 1:  # 平多單
            exit_price = price - slippage_cost  # 出場價格減去滑價
            profit = exit_price - entry_price # 扣除雙邊滑價成本
            trades.append((indices[i], dt[i], 'Long_Exit', exit_price, profit))
            position = 0

        # 空單邏輯
        if position == 0 and short_entry[i] == 1:  # 進空單
            position = -1
            entry_price = price - slippage_cost  # 進場價格減去滑價
            trades.append((indices[i], dt[i], 'Short_Entry', entry_price))
        elif position == -1 and short_exit[i] == 1:  # 平空單
            exit_price = price + slippage_cost  # 出場價格加上滑價
            profit = entry_price - exit_price # 扣除雙邊滑價成本
            trades.append((indices[i], dt[i], 'Short_Exit', exit_price, profit))
            position = 0

    return trades

def process_trades_to_df(trades):
    """
    處理交易記錄，過濾未配對的交易，並計算持倉時間。

    Parameters:
        trades (list): 原始交易資料，進場交易需有 4 個元素，出場交易需有 5 個元素。

    Returns:
        pd.DataFrame: 包含持倉時間的交易資料。
    """
    # 將交易記錄轉為 DataFrame
    columns = ['index', 'datetime', 'trade_type', 'price']
    entry_trades = [trade for trade in trades if len(trade) == 4]  # 進場交易
    exit_trades = [trade for trade in trades if len(trade) == 5]  # 出場交易
    entry_df = pd.DataFrame(entry_trades, columns=columns)  # 進場不包含 profit
    exit_df = pd.DataFrame(exit_trades, columns=columns + ['profit'])  # 出場包含 profit
    if len(entry_df) == len(exit_df)==0:
        return pd.DataFrame()
    dfs = [entry_df, exit_df]
    dfs = [d for d in dfs if not d.empty]  # 過濾掉空的
    if dfs:  # 至少有一個非空
        df = pd.concat(dfs).sort_values('index')
    else:
        return pd.DataFrame()
    df = df.sort_index()
    df = df.sort_values(['index','datetime'])

    # 分組處理交易
    long_entries = df[df['trade_type'] == 'Long_Entry']
    long_exits = df[df['trade_type'] == 'Long_Exit']
    short_entries = df[df['trade_type'] == 'Short_Entry']
    short_exits = df[df['trade_type'] == 'Short_Exit']

    # 處理 Long 交易
    len_long_entries = len(long_entries)
    len_long_exits = len(long_exits)
    if len_long_entries == len_long_exits:
        # print(f"✅ Long_Entry 和 Long_Exit 數量{len_long_entries}相等，無需處理。")
        pass
    else:
        # print(f"⚠️ Long_Entry 和 Long_Exit 數量不同，開始處理...")
        if len_long_entries > len_long_exits:
            long_entries = long_entries.iloc[:-1]
            # print(f"已刪除 Long_Entry 的最後一行，現在大小為 {len(long_entries)}")
        elif len_long_exits > len_long_entries:
            long_exits = long_exits.iloc[:-1]
            print(f"❌已刪除 Long_Exit 的最後一行，現在大小為 {len(long_exits)} 不應該出現此錯誤")
        if len(long_entries) != len_long_exits:
            raise ValueError(f"❌ 處理後 Long_Entry 和 Long_Exit 數量仍然不相等！目前大小："
                            f"Long_Entry = {len(long_entries)}, Long_Exit = {len(long_exits)}")
        else:
            # print(f"✅ 處理完成，現在 Long_Entry 和 Long_Exit 數量{len_long_entries}相等。")
            pass

    # 處理 Short 交易
    len_short_entries = len(short_entries)
    len_short_exits = len(short_exits)
    if len_short_entries == len_short_exits:
        # print(f"✅ Short_Entry 和 Short_Exit 數量{len_short_entries}相等，無需處理。")
        pass
    else:
        # print(f"⚠️ Short_Entry 和 Short_Exit 數量不同，開始處理...")
        if len_short_entries > len_short_exits:
            short_entries = short_entries.iloc[:-1]
            # print(f"已刪除 Short_Entry 的最後一行，現在大小為 {len(short_entries)}")
        elif len_short_exits > len_short_entries:
            short_exits = short_exits.iloc[:-1]
            print(f"❌已刪除 Short_Exit 的最後一行，現在大小為 {len(short_exits)} 不應該出現此錯誤")
        if len(short_entries) != len_short_exits:
            raise ValueError(f"❌ 處理後 Short_Entry 和 Short_Exit 數量仍然不相等！目前大小："
                            f"Short_Entry = {len(short_entries)}, Short_Exit = {len(short_exits)}")
        else:
            # print(f"✅ 處理完成，現在 Short_Entry 和 Short_Exit 數量{len_short_entries}相等。")
            pass

    dfs = [long_entries, long_exits, short_entries, short_exits]
    dfs = [d for d in dfs if not d.empty]  # 過濾掉空的
    if dfs:  # 至少有一個非空
        df = pd.concat(dfs).sort_index()
    else:
        return pd.DataFrame()
    df = df.reset_index(names='original_index')
    df = df.sort_values(['datetime','original_index'])
    df['check'] = 0  # 預設為 0，表示無問題
    df['next_trade_type'] = df['trade_type'].shift(-1)  # 獲取下一行的交易類型
    df.loc[(df['trade_type'] == 'Long_Entry') & (df['next_trade_type'] != 'Long_Exit'), 'check'] = 1
    df.loc[(df['trade_type'] == 'Short_Entry') & (df['next_trade_type'] != 'Short_Exit'), 'check'] = 1
    if len(df[df['check'] == 1]) ==0:
        # print('資料確認無錯誤配對')
        pass
    df = df.drop(columns=['next_trade_type','check'])
    df['profit_percent'] = df['profit'] / df['price'].shift(1)
    df['profit_percent'] = df['profit_percent']
    df['hold_time'] = df['datetime'] - df['datetime'].shift(1)
    df['hold_ticks'] = df['index'] - df['index'].shift(1)
    df.loc[df['trade_type'].str.contains('Entry'), 'hold_time'] = None
    df.loc[df['trade_type'].str.contains('Entry'), 'hold_ticks'] = None

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    return df
         
def calculate_stats(df):
    '''
    統計交易績效指標。df 為交易明細（每筆進出都一列）
    '''

    # 1. 交易基本統計
    trades = len(df)  # 交易紀錄總數（進場＋出場筆數）
    if trades <= 1:   # 若無法配對成一筆交易，直接回傳空
        return {}

    trades_long = len(df[df['trade_type'].isin(['Long_Entry', 'Long_Exit'])])  # 多單交易次數

    # 2. 勝率
    win_trades = len(df[df['profit'] > 0])       # 賺錢的出場次數
    win_rate = (win_trades / (trades/2))         # 勝率，分母用交易pair數

    # 3. 盈虧比（Profit Factor）
    total_profit_percent = df[df['profit_percent'] > 0]['profit_percent'].sum()   # 總獲利%
    total_loss_percent = df[df['profit_percent'] < 0]['profit_percent'].sum()     # 總虧損%
    profit_factor = total_profit_percent / abs(total_loss_percent) if abs(total_loss_percent) > 0 else float('inf')

    # 4. 多單總獲利
    profit_long = df[df['trade_type'] == 'Long_Exit'].profit_percent.sum()        # 多單出場的總獲利%

    # 5. 累積績效曲線
    df['cumulative_profit_percent'] = df['profit_percent'].cumsum()               # 累積損益曲線

    # 6. 最大回撤
    df['peak_percent'] = df['cumulative_profit_percent'].cummax()                 # 歷史高點
    df['drawdown_percent'] = df['cumulative_profit_percent'] - df['peak_percent'] # 當前距離高點跌幅
    max_drawdown_percent = df['drawdown_percent'].min()                           # 取最大下跌（負值）

    # 7. 月、年統計
    monthly_profit = df.groupby(['year', 'month'])['profit_percent'].sum()
    yearly_profit = df.groupby('year')['profit_percent'].sum()

    # 8. 月績效衍生指標
    positive_months = len(monthly_profit[monthly_profit > 0])                     # 盈利月份數
    total_months = len(monthly_profit)
    positive_month_proportion = (positive_months / total_months) if total_months > 0 else 0  # 正收益月比例

    negative_months = len(monthly_profit[monthly_profit < 0])                     # 虧損月份數
    total_negative_loss = monthly_profit[monthly_profit < 0].sum()
    avg_loss_negative_months = (total_negative_loss / negative_months) if negative_months > 0 else 0  # 虧損月平均

    # 9. 報酬極值
    gross_profit_percent = df['profit_percent'].sum()                             # 全期間總損益
    monthly_max_profit = monthly_profit.max()                                     # 月最大損益
    monthly_min_profit = monthly_profit.min()                                     # 月最小損益
    year_max_profit = yearly_profit.max()                                         # 年最大損益
    year_min_profit = yearly_profit.min()                                         # 年最小損益

    # 10. 報酬波動
    pair_profit_avg = df['profit_percent'].mean()                                # 平均每筆
    pair_profit_std_dev = df['profit_percent'].std()                                   # 標準差

    # 11. 年化複利績效
    first_time = pd.to_datetime(df['datetime'].min())
    last_time = pd.to_datetime(df['datetime'].max())
    years = (last_time - first_time).days / 365.25
    if years <= 0 or (1 + gross_profit_percent) <= 0:
        CAGR = float('nan')
    else:
        CAGR = (1 + gross_profit_percent) ** (1 / years) - 1

    # 12. 年化 Sharpe、Calmar
    total_pairs = trades/2
    trades_per_year = total_pairs / years if years > 0 else 0
    sharpe_ann = (pair_profit_avg / pair_profit_std_dev) * np.sqrt(trades_per_year) if pair_profit_std_dev > 0 and trades_per_year > 0 else 0

    max_drawdown = abs(max_drawdown_percent)
    calmar_ann = CAGR / max_drawdown if max_drawdown > 0 else 0

    # 13. 統整回傳
    return {
        # 風險
        'max_drawdown': max_drawdown_percent,           # 最大回撤（負值）
        # 收益
        'gross_profit': gross_profit_percent,           # 全期總報酬率
        'profit_long': profit_long,                     # 多單總報酬率
        'pair_profit_avg': pair_profit_avg,             # 平均交易對(pair)報酬
        # 波動
        'pair_profit_std_dev': pair_profit_std_dev,     # 報酬標準差
        # 年化績效
        'CAGR': CAGR,                                   # 年化複利(%)
        'sharpe_ann': sharpe_ann,                       # 年化Sharpe
        'calmar_ann': calmar_ann,                       # Calmar
        # 頻率&穩健性
        'trades': trades,                               # 交易次數
        'trades_long': trades_long,                     # 多單次數
        'profit_factor': profit_factor,                 # 盈虧比
        'win_rate': win_rate,                           # 勝率 (0~1)
        'positive_month_proportion': positive_month_proportion,   # 正報酬月比例
        'avg_loss_negative_months': avg_loss_negative_months,     # 虧損月均虧損(%)（負值）
        # 報酬極值
        'month_max_profit': monthly_max_profit,
        'month_min_profit': monthly_min_profit,
        'year_max_profit': year_max_profit,
        'year_min_profit': year_min_profit,
    }

# 報表模組
def Report_Year_month(trades_df, output_csv='Backtest_Report.csv'):
    """
    輸入交易 DataFrame，輸出包含每年與每月收益統計的 CSV 文件。

    Parameters:
    trades_df (pd.DataFrame): 包含交易數據的 DataFrame。
    output_csv (str): 輸出的 CSV 文件名稱，默認為 'Combined_Trade_Summary.csv'。

    Returns:
    None
    """
    if trades_df.empty:
        print("無交易可生成報告")
        return
    # 每年的多單收益和交易數據
    yearly_long_stats = trades_df[trades_df['trade_type'] == 'Long_Exit']
    yearly_long_summary = yearly_long_stats.groupby('year').agg(
        long_profit=('profit_percent', 'sum'),  # 多單收益總和
        long_trades=('profit_percent', 'count'),  # 多單交易筆數
    ).reset_index()

    yearly_long_summary['long_profit'] = yearly_long_summary['long_profit'].round(2)

    # 獲取每年的總收益和總交易數據
    yearly_total_stats = trades_df.groupby('year').agg(
        total_profit=('profit_percent', 'sum'),  # 總收益
        total_trades=('profit_percent', 'count'),  # 總交易筆數
    ).reset_index()

    yearly_total_stats['total_profit'] = yearly_total_stats['total_profit'].round(2)

    # 合併多單收益與總收益數據
    yearly_summary = pd.merge(yearly_total_stats, yearly_long_summary, on='year', how='left')

    yearly_summary = yearly_summary[['year', 'total_profit', 'total_trades', 'long_profit', 'long_trades']]

    # 每月的多單收益和交易數據
    monthly_long_stats = trades_df[trades_df['trade_type'] == 'Long_Exit']
    monthly_long_summary = monthly_long_stats.groupby(['year', 'month']).agg(
        long_profit=('profit_percent', 'sum'),  # 多單收益總和
        long_trades=('profit_percent', 'count'),  # 多單交易筆數
    ).reset_index()

    monthly_long_summary['long_profit'] = monthly_long_summary['long_profit'].round(2)

    # 獲取每月的總收益和總交易數據
    monthly_total_stats = trades_df.groupby(['year', 'month']).agg(
        total_profit=('profit_percent', 'sum'),  # 總收益
        total_trades=('profit_percent', 'count'),  # 總交易筆數
    ).reset_index()

    monthly_total_stats['total_profit'] = monthly_total_stats['total_profit'].round(2)

    # 合併多單收益與總收益數據
    monthly_summary = pd.merge(monthly_total_stats, monthly_long_summary, on=['year', 'month'], how='left')

    monthly_summary = monthly_summary[['year', 'month', 'total_profit', 'total_trades', 'long_profit', 'long_trades']]

    # 合併年和月數據
    yearly_summary['month'] = 'ALL'  # 標記全年數據
    yearly_summary['month'] = yearly_summary['month'].astype(str)
    monthly_summary['month'] = monthly_summary['month'].astype(str)

    combined_summary = pd.concat([yearly_summary, monthly_summary], ignore_index=True)

    # 保存為單一 CSV
    combined_summary.to_csv(output_csv, index=False)

    print(f"Combined yearly and monthly summaries saved to '{output_csv}'")

# 成功 訊號策略區 
def generate_signals_1_lot(df, threshold_1=43, threshold_2=150):
    '''
    threshold_1 = 0 為 結算日的13:01 進場多單 best 43  13:44 進場空單 best 87 15:44
    threshold_2 = 1 為 結算日的13:00 出場多單 best 150 10:31 出場空單 best 15 12:46
    '''
    df['long_entry'] = 0
    df['long_exit'] = 0
    df['short_entry'] = 0
    df['short_exit'] = 0
    grouped = df.groupby('duedmonth')
    # 遍歷每個組別，設定進場與出場點
    for name, group in grouped:
        # 設定進場點：該月的第一筆資料
        df.loc[group.index[threshold_1], 'long_entry'] = 1
        # 設定出場點：該月的最後一筆資料
        df.loc[group.index[-threshold_2], 'long_exit'] = 1    
    return df

def generate_signals_ai_20250616_164528_4(df, bb_window=18250, bb_std_dev_multiplier=9.79, mfi_window=15846, atr_window=17940, atr_norm_window=9489, vol_stability_threshold=9.41):
    bb_std_dev_multiplier_scaled = bb_std_dev_multiplier / 10.0
    vol_stability_threshold_scaled = vol_stability_threshold / 10.0

    df['bb_upper'], df['bb_middle'], df['bb_lower'] = indicator_bollinger_bands(df['price'], bb_window, bb_std_dev_multiplier_scaled)
    df['mfi'] = indicator_mfi(df['cap'], df['price'], mfi_window)
    df['atr_norm'] = indicator_atr_norm(df['price'], atr_window, atr_norm_window)

    df['is_stable_volatility'] = (df['atr_norm'].abs() < vol_stability_threshold_scaled).astype(int)

    df['long_entry'] = 0
    df['long_exit'] = 0
    df['short_entry'] = 0
    df['short_exit'] = 0

    df['long_entry'] = (
        (df['price'] > df['bb_upper']) &
        (df['price'].shift(1) <= df['bb_upper'].shift(1)) &
        (df['mfi'] > 50) &
        (df['mfi'] > df['mfi'].shift(1)) &
        (df['is_stable_volatility'] == 1)
    ).astype(int)

    df['short_entry'] = (
        (df['price'] < df['bb_lower']) &
        (df['price'].shift(1) >= df['bb_lower'].shift(1)) &
        (df['mfi'] < 50) &
        (df['mfi'] < df['mfi'].shift(1)) &
        (df['is_stable_volatility'] == 1)
    ).astype(int)

    df['long_exit'] = (
        ((df['price'] < df['bb_middle']) & (df['price'].shift(1) >= df['bb_middle'].shift(1))) |
        ((df['mfi'] < 50) & (df['mfi'].shift(1) >= 50))
    ).astype(int)

    df['short_exit'] = (
        ((df['price'] > df['bb_middle']) & (df['price'].shift(1) <= df['bb_middle'].shift(1))) |
        ((df['mfi'] > 50) & (df['mfi'].shift(1) <= 50))
    ).astype(int)

    df = df.drop(columns=['bb_upper', 'bb_middle', 'bb_lower', 'mfi', 'atr_norm', 'is_stable_volatility'])

    return df

# 失敗 訊號策略區
def generate_signals_ai_20250617_221222_9(df, rsi_period=14, mfi_period=14, atr_period=14, divergence_lookback_bars=20, rsi_entry_level_long=30, rsi_entry_level_short=70, atr_max_pct_of_price=0.015):
    df['rsi'] = indicator_rsi(df['price'], rsi_period)
    df['mfi'] = indicator_mfi(df['cap'], df['price'], mfi_period)
    df['atr'] = indicator_atr(df['price'], atr_period)

    df['volatility_filter'] = (df['atr'] / df['price'] < atr_max_pct_of_price).astype(int)

    df['bullish_divergence'] = ((df['price'] < df['price'].shift(divergence_lookback_bars)) & \
                                (df['rsi'] > df['rsi'].shift(divergence_lookback_bars))).astype(int)

    df['bearish_divergence'] = ((df['price'] > df['price'].shift(divergence_lookback_bars)) & \
                                (df['rsi'] < df['rsi'].shift(divergence_lookback_bars))).astype(int)

    df['mfi_rising'] = (df['mfi'].diff() > 0).astype(int)
    df['mfi_falling'] = (df['mfi'].diff() < 0).astype(int)

    df['long_entry'] = 0
    df['long_exit'] = 0
    df['short_entry'] = 0
    df['short_exit'] = 0

    df['long_entry'] = (
        (df['bullish_divergence'] == 1) &
        (df['rsi'] <= rsi_entry_level_long) &
        (df['mfi_rising'] == 1) &
        (df['volatility_filter'] == 1)
    ).astype(int)

    df['long_exit'] = (
        (df['rsi'] >= rsi_entry_level_short) |
        (df['mfi_falling'] == 1)
    ).astype(int)

    df['short_entry'] = (
        (df['bearish_divergence'] == 1) &
        (df['rsi'] >= rsi_entry_level_short) &
        (df['mfi_falling'] == 1) &
        (df['volatility_filter'] == 1)
    ).astype(int)

    df['short_exit'] = (
        (df['rsi'] <= rsi_entry_level_long) |
        (df['mfi_rising'] == 1)
    ).astype(int)

    return df


'''
策略設計模版
def generate_signals_template(df, ...):
    # 前置指標計算
    df['指標1'] = ...  # 例如 WR
    df['指標2'] = ...  # 例如 True Cap
    df['指標3'] = ...  # 例如 cap_zscore

    # 多因子盤型判斷
    df['is_trend'] = ...
    df['is_range'] = ...

    # 進出場邏輯設計
    df['long_entry'] = ((df['指標1'] ... ) & (df['is_trend'])).astype(int)
    df['long_exit'] = ...
    # 其他訊號

    # 出場一定包含 止盈/止損/時間
    # 填補/forward fill所有 entry/exit price

    # 持倉期間結構與避免反覆進出
    ...

    return df

def generate_signals_multi_factor_weighted(df, ...):
    """
    多指標同時納入，並設置加權分數，超過分數門檻才進場。
    """
    # Example: WR, RSI, True Cap, ATR, Z-Score
    df['wr'] = indicator_williams_r(df['price'], period1)
    df['rsi'] = indicator_rsi(df['price'], period2)
    df['true_cap'] = indicator_true_cap_moving_average(df['cap'], df['volume'], period3)
    df['cap_zscore'] = ((df['cap'] - df['cap'].rolling(20).mean()) / df['cap'].rolling(20).std())

    df['score'] = (
        (df['wr'] < 20) * 1 +
        (df['rsi'] < 30) * 1 +
        (df['price'] > df['true_cap']) * 1 +
        (df['cap_zscore'] > 2) * 1
    )

    df['long_entry'] = (df['score'] >= 3).astype(int)  # 3個條件以上才進場
    # 出場可再根據個別條件設計
    ...
    return df

def generate_signals_regime_filter(df, ...):
    """
    先判斷是否為趨勢盤/盤整盤，再決定執行哪種策略（ex: ATR, Z-Score, 布林等）。
    """
    # 盤型判斷（ex: True Cap斜率或波動率）
    df['tc_slope'] = df['true_cap'].diff(window)
    df['is_trend'] = (df['tc_slope'].abs() > 某門檻).astype(int)
    df['is_range'] = (df['tc_slope'].abs() < 某門檻).astype(int)

    # 趨勢盤用追突破策略，盤整盤用反轉策略
    df['long_entry'] = 0
    df.loc[df['is_trend'] & (df['price'] > df['upper_band']), 'long_entry'] = 1
    df.loc[df['is_range'] & (df['price'] < df['lower_band']), 'long_entry'] = 1
    ...
    return df

def generate_signals_multi_timeframe(df, df_higher_tf, ...):
    """
    長短周期訊號疊加（ex: 1分鐘 + 15分鐘），高週期多頭才允許低週期多單進場。
    """
    # 高週期均線
    df_higher_tf['ma_high'] = df_higher_tf['price'].rolling(window=高週期).mean()
    df = df.merge(df_higher_tf[['datetime', 'ma_high']], on='datetime', how='left')
    df['long_entry'] = ((df['price'] > df['true_cap']) & (df['price'] > df['ma_high'])).astype(int)
    ...
    return df

def generate_signals_event_confirm(df, ...):
    """
    當日某事件（ex: 開盤跳空、急殺/急拉、訊號異常量能）發生後，搭配後續結構確認才允許進出場。
    """
    # 例如：開盤前5分鐘大跳空 + 首次觸及WR極值
    df['jump_open'] = ((df['price'].pct_change(periods=5).abs() > 0.01)).astype(int)
    df['long_entry'] = ((df['jump_open'] == 1) & (df['wr'] < 10) & (df['price'] > df['true_cap'])).astype(int)
    ...
    return df

def generate_signals_order_flow(df, ...):
    """
    資金流/量能變化領先訊號設計，例如cap暴增或累積買單激增時才進場。
    """
    df['cap_mean'] = df['cap'].rolling(60).mean()
    df['cap_spike'] = ((df['cap'] > df['cap_mean'] * 2)).astype(int)
    df['long_entry'] = ((df['cap_spike'] == 1) & (df['price'] > df['true_cap'])).astype(int)
    ...
    return df

def add_trailing_stop_logic(df, ...):
    """
    可直接加在任何策略的df上，自動加上移動停利和固定停損。
    """
    # 假設 entry_price 已計算
    df['stop_loss_long'] = df['entry_price'] * (1 - 固定停損百分比)
    df['trailing_high'] = df.groupby('long_entry'.cumsum())['price'].cummax()
    df['trailing_stop_long'] = df['trailing_high'] * (1 - 移動停利百分比)
    df['long_exit'] = ((df['price'] <= df['stop_loss_long']) | (df['price'] <= df['trailing_stop_long'])).astype(int)
    ...
    return df

'''
