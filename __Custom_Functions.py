import shioaji as sj
import pandas as pd
import requests
import time 
import datetime
import QuantLib as ql
from shioaji import BidAskFOPv1, Exchange
import logging

# 登錄 API，並 Ping 203.66.91.161
def api_login(account_num, simulation, subscribe_trade):
    """
    登錄 Shioaji API 並啟用憑證。

    :param account_num: 指定使用第幾個帳號 API 資料
    :param simulation: 是否為模擬模式 (True/False)
    :param subscribe_trade: 是否訂閱交易回報
    :return: 登錄成功的 API 物件
    """
    # 是否測試模式
    api = sj.Shioaji(simulation=simulation)

    # 讀取 API KEY 的 CSV 文件
    df = pd.read_csv('Api_SJ.csv')  # 檔案名稱為 Api_SJ.csv
    APIs_column = account_num       # 指定要使用的帳號資料 (第幾列)

    # 執行 API 登錄
    api.login(
        df.iloc[APIs_column].Api_key1,      # API Key 1
        df.iloc[APIs_column].Api_key2,      # API Key 2
        subscribe_trade=subscribe_trade,    # 是否訂閱交易回報，顯示於 Prompt
        contracts_timeout=30000             # 設定合約超時時間
    )

    # 憑證處理
    api.activate_ca(
        ca_path='SJ_Ca.pfx',                # 憑證檔案路徑
        ca_passwd=df.iloc[APIs_column].ca_passwd,  # 憑證密碼
        person_id=df.iloc[APIs_column].person_id   # 使用者身份證字號
    )

    # 返回 API 物件
    return api

# 找期貨合約
def get_future_contract(api,symbol,contract_yearmonth):
    return api.Contracts.Futures[symbol][symbol+contract_yearmonth]

# 找選擇權合約
def get_option_contract(api,symbol,contract_yearmonth,strike,callput):
    return api.Contracts.Options[symbol][symbol+contract_yearmonth+strike+callput]

# LINE 饅頭說~
def mantal_says_line(message,APIs_column):
    logging.basicConfig(filename='mantal_says_line.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    df=pd.read_csv('Api_Line.csv')
    # var 讀取第幾個API
    APIs_column = APIs_column
    Line_Notify_Account = {'Client ID': df.iloc[APIs_column].Client_ID,
                           'Client Secret': df.iloc[APIs_column].Client_Secret,
                           'token': df.iloc[APIs_column].token}
    headers = {'Authorization': 'Bearer ' + Line_Notify_Account['token'],
               "Content-Type": "application/x-www-form-urlencoded"}
    params = {"message": message}
    try:
        requests.post("https://notify-api.line.me/api/notify", headers=headers, params=params)
    except Exception as e:
        logging.info(f'{e}') ##########

# Telegrame 饅頭說~
def mantal_says_telegram(message,APIs_column):
    logging.basicConfig(filename='mantal_says_telegram.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    df=pd.read_csv('Api_Telegram.csv')
    # var 讀取第幾個API
    APIs_column=APIs_column
    TOKEN = df.iloc[APIs_column].TOKEN
    chat_id = df.iloc[APIs_column].chat_id
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    try:
        requests.get(url)
    except Exception as e:
        logging.info(f'{e}')

# 獲得最新大台snapshot # 非交易時間或是取得錯誤都會回報 'NoneType' object has no attribute 'code'
# 合約都是在每天早上八點更新 fetch_contract
def get_latest_TXF_snapshot(api):
    # 大台近月次月處理
    futures=list(api.Contracts.Futures.TXF)
    sorted_contracts = sorted([contract for contract in futures if not contract.code.startswith('TXFR')], key=lambda x: x.symbol[-6:])
    delivery_date = datetime.datetime.strptime(sorted_contracts[0].delivery_date, '%Y/%m/%d')
    update_date = datetime.datetime.strptime(sorted_contracts[0].update_date, '%Y/%m/%d')
    # 日盤 夜盤定義
    now = datetime.datetime.now().time()
    # start_time1 = datetime.time(8, 28, 0)
    # end_time1 = datetime.time(14, 0, 0)
    start_time2 = datetime.time(13, 30, 00)
    end_time2 = datetime.time(8, 00, 00)
    # 更新日期等於到期日代表結算日當天 更新日期大於交割日期代表結算日後移一個交易日
    # 更新日期大部分時間應該要小於結算日期，只有結算日會等於更新日期 颱風天剛好禮拜三delivery_date 不會更新
    # 如果更新日期大於結算日期代表結算日期沒被更新，結算日就變成更新日，就算有更新也會更新日期等於結算日期，雙重解決
    # 交割日期=資料更新日期=今天是結算日
    # update_date就算過0點也不會更新成下一日因為合約是每天早上8點刷新
    # 颱風假10/2 結算日10/2 當天並不會更新delivery 10/3再放也不更新 直到10/4正常開盤 早上八點合約更新後會顯示10/4的delivery
    # 結算日12點30分就切換合約
    if delivery_date <= update_date and (start_time2 <= now or now <= end_time2):
        # 這原本是0,1 但是結算日的夜盤會依然留存早上剛結算的月合約，所以要改成1,2
        near_month = sorted_contracts[1].delivery_month
        # next_month = sorted_contracts[2].delivery_month
        # 取得近月合約最新報價
        return api.snapshots([get_future_contract(api,'TXF',near_month)])[0]
    # 正常時段調用
    else:
        near_month = sorted_contracts[0].delivery_month
        # next_month = sorted_contracts[1].delivery_month
        # 取得近月合約最新報價
        return api.snapshots([get_future_contract(api,'TXF',near_month)])[0]

def get_next_TXF_snapshot(api):
    # 大台近月次月處理
    futures=list(api.Contracts.Futures.TXF)
    sorted_contracts = sorted([contract for contract in futures if not contract.code.startswith('TXFR')], key=lambda x: x.symbol[-6:])
    delivery_date = datetime.datetime.strptime(sorted_contracts[0].delivery_date, '%Y/%m/%d')
    update_date = datetime.datetime.strptime(sorted_contracts[0].update_date, '%Y/%m/%d')
    # 日盤 夜盤定義
    now = datetime.datetime.now().time()
    # start_time1 = datetime.time(8, 28, 0)
    # end_time1 = datetime.time(14, 0, 0)
    start_time2 = datetime.time(13, 30, 00)
    end_time2 = datetime.time(8, 00, 00)
    # 更新日期等於到期日代表結算日當天 更新日期大於交割日期代表結算日後移一個交易日
    # 更新日期大部分時間應該要小於結算日期，只有結算日會等於更新日期 颱風天剛好禮拜三delivery_date 不會更新
    # 如果更新日期大於結算日期代表結算日期沒被更新，結算日就變成更新日，就算有更新也會更新日期等於結算日期，雙重解決
    # 交割日期=資料更新日期=今天是結算日
    # update_date就算過0點也不會更新成下一日因為合約是每天早上8點刷新
    # 颱風假10/2 結算日10/2 當天並不會更新delivery 10/3再放也不更新 直到10/4正常開盤 早上八點合約更新後會顯示10/4的delivery
    # 結算日12點30分就切換合約
    if delivery_date <= update_date and (start_time2 <= now or now <= end_time2):
        # 這原本是0,1 但是結算日的夜盤會依然留存早上剛結算的月合約，所以要改成1,2
        # near_month = sorted_contracts[1].delivery_month
        next_month = sorted_contracts[2].delivery_month
        # 取得近月合約最新報價
        return api.snapshots([get_future_contract(api,'TXF',next_month)])[0]
    # 正常時段調用
    else:
        # near_month = sorted_contracts[0].delivery_month
        next_month = sorted_contracts[1].delivery_month
        # 取得近月合約最新報價
        return api.snapshots([get_future_contract(api,'TXF',next_month)])[0]

def get_next_week_TXO_code(api):
    """
    獲取下一週的 TXO 合約。
    
    :param api: 交易 API 接口物件
    :return: TXO 合約代碼 (格式如 TX4X4)
    """
    options = []  # 儲存所有選擇權合約
    possible_codes = [1, 2, 'O', 4, 5]  # 所有可能的周月選代碼

    # 遍歷所有可能的代碼，獲取對應的選擇權合約
    for code in possible_codes:
        try:
            option_list = list(getattr(api.Contracts.Options, f'TX{code}'))
            options.extend(option_list)
        except AttributeError:
            continue  # 如果代碼不存在，直接跳過

    # 提取所有合約的到期日並排序
    unique_delivery_dates = sorted({option.delivery_date for option in options})

    # 獲取最新一週與下一週的到期日
    latest_week_TXO_date = unique_delivery_dates[0]
    next_week_TXO_date = unique_delivery_dates[1]

    # 過濾出對應到期日的合約
    latest_week_TXO_contract = [
        option for option in options if option.delivery_date == latest_week_TXO_date
    ][0]
    next_week_TXO_contract = [
        option for option in options if option.delivery_date == next_week_TXO_date
    ][0]

    # 判斷是否接近結算日
    delivery_date = datetime.datetime.strptime(latest_week_TXO_contract.delivery_date, '%Y/%m/%d')
    update_date = datetime.datetime.strptime(latest_week_TXO_contract.update_date, '%Y/%m/%d')
    now = datetime.datetime.now().time()

    settlement_start_time = datetime.time(12, 29, 59)
    settlement_end_time = datetime.time(7, 59, 59)

    if delivery_date <= update_date and (settlement_start_time <= now or now <= settlement_end_time): 
        # 結算日當天返回下一週 因為本周的合約要等到明天才會消失
        return next_week_TXO_contract.code[:3] + next_week_TXO_contract.code[-2:]
    else:
        # 返回最新一週的合約代碼
        return latest_week_TXO_contract.code[:3] + latest_week_TXO_contract.code[-2:]

def get_next_next_week_TXO_code(api):
    """
    獲取下下週的 TXO 合約。
    
    :param api: 交易 API 接口物件
    :return: TXO 合約代碼 (格式如 TX4X4)
    """
    options = []  # 儲存所有選擇權合約
    possible_codes = [1, 2, 'O', 4, 5]  # 所有可能的周月選代碼

    # 遍歷所有可能的代碼，獲取對應的選擇權合約
    for code in possible_codes:
        try:
            option_list = list(getattr(api.Contracts.Options, f'TX{code}'))
            options.extend(option_list)
        except AttributeError:
            continue  # 如果代碼不存在，直接跳過

    # 提取所有合約的到期日並排序
    unique_delivery_dates = sorted({option.delivery_date for option in options})

    # 獲取下一週與下下週的到期日
    next_week_TXO_date = unique_delivery_dates[1]
    next_next_week_TXO_date = unique_delivery_dates[2]

    # 過濾出對應到期日的合約
    next_week_TXO_contract = [
        option for option in options if option.delivery_date == next_week_TXO_date
    ][0]
    next_next_week_TXO_contract = [
        option for option in options if option.delivery_date == next_next_week_TXO_date
    ][0]

    # 判斷是否接近結算日
    delivery_date = datetime.datetime.strptime(next_week_TXO_contract.delivery_date, '%Y/%m/%d')
    update_date = datetime.datetime.strptime(next_week_TXO_contract.update_date, '%Y/%m/%d')
    now = datetime.datetime.now().time()

    settlement_start_time = datetime.time(12, 29, 59)
    settlement_end_time = datetime.time(7, 59, 59)

    if delivery_date <= update_date and (settlement_start_time <= now or now <= settlement_end_time):
        # 返回下下週的合約代碼
        return next_next_week_TXO_contract.code[:3] + next_next_week_TXO_contract.code[-2:]
    else:
        # 返回下一週的合約代碼
        return next_week_TXO_contract.code[:3] + next_week_TXO_contract.code[-2:]

# 取消所有委託單
def cancel_all_orders(api):
    try:
        api.update_status(api.futopt_account)
        trade_list = api.list_trades()
        cancel_count = 0

        for trade in trade_list:
            status_name = trade.status.status.name
            if status_name in ['PreSubmitted', 'Submitted', 'PartFilled']:
                api.cancel_order(trade)
                print(f"\033[1;34m[INFO]\033[0m Canceled: {trade.contract.code} Status: {status_name}")
                cancel_count += 1

        if cancel_count == 0:
            print(f"\033[1;34m[INFO]\033[0m No orders to cancel")
        else:
            print(f"\033[1;34m[INFO]\033[0m Total canceled: {cancel_count}")

    except Exception as e:
        print(f"\033[1;33m[WARN]\033[0m Failed to cancel orders: {e}")

# 取消所有委託單 v2
def cancel_all_orders_v2(api):
    api.update_status(api.futopt_account)
    cancel_status = False
    for idx,t in enumerate(api.list_trades()):
        if t.status.status in [sj.constant.Status.PreSubmitted,sj.constant.Status.Submitted,sj.constant.Status.PartFilled] :
            api.cancel_order(t,timeout=0)
            print('成功刪單')
            cancel_status = True
    if cancel_status != True:
        print('無刪單執行')
        
# 回傳所有未成交委託回報
def get_all_order_df(api):
    df=pd.DataFrame()
    #更新狀態
    api.update_status(api.futopt_account)
    #找出所有委託
    trade_list=api.list_trades()
    for i in trade_list:
        name=i.status.status.name
        #如果委託是未成交或部分成交的
        if (name == 'Submitted') or (name == 'filling'):
            data = {'symbol': [i.contract.symbol],'delivery_date': [i.contract.delivery_date],'reference': [i.contract.reference],
            'action': [i.order.action],'price': [i.order.price],'ordno': [i.order.ordno],'price_type': [i.order.price_type],
            'order_type': [i.order.order_type],'octype': [i.order.octype],'status': [i.status.status],
            'status_code': [i.status.status_code],'order_datetime': [i.status.order_datetime],'modified_price': [i.status.modified_price],
            }
            df_single = pd.DataFrame(data)
            df=pd.concat([df,df_single])         
    return df

# 計算選擇權指標  最後一天會出錯 option expired  先以延後一天到期處理
# maturity_date=到期日 spot_price=市價 strike_price=履約價 annual_volatility=標的年波動率 risk_free_rate=市場無風險利率 dividend_rate=標的利率 option_type=買權賣權 marketprice
def ql_option_metrics_calculator(maturity_date, spot_price, strike_price, annual_volatility, risk_free_rate,
                                 dividend_rate, option_type, marketprice):
    price = delta = theta = impliedVolatility = 0
    try:
        payoff = ql.PlainVanillaPayoff(option_type, strike_price)
        exercise = ql.EuropeanExercise(maturity_date)
        underlying = ql.SimpleQuote(spot_price)
        volatility_quote = ql.SimpleQuote(annual_volatility)
        risk_free_curve = ql.FlatForward(0, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual360())
        dividend_curve = ql.FlatForward(0, ql.TARGET(), ql.QuoteHandle(ql.SimpleQuote(dividend_rate)), ql.Actual360())
        process = ql.BlackScholesMertonProcess(ql.QuoteHandle(underlying), ql.YieldTermStructureHandle(dividend_curve),
                                                ql.YieldTermStructureHandle(risk_free_curve),
                                                ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, ql.TARGET(),
                                                                                                    ql.QuoteHandle(
                                                                                                        ql.SimpleQuote(
                                                                                                            annual_volatility)),
                                                                                                    ql.Actual360())))
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        price = option.NPV()
        delta = option.delta()
        theta = option.theta() / 365.25 # 這個365感覺可以改成360 可以觀察差異
        impliedVolatility = option.impliedVolatility(marketprice, process)
    except:
        pass
    return price, delta, theta, impliedVolatility

# 確認目前是否開盤
def check_market_open(api,timeout):
    # CALLBACK 接收器
    market_is_open_or_not=False
    def bidask_callback_function_for_check(exchange: Exchange, bidask: BidAskFOPv1):
        nonlocal market_is_open_or_not
        try:
            if  len(bidask['code']) > 0 :# 目前為交易時段
                market_is_open_or_not=True
        except:
            market_is_open_or_not=False # 目前非交易時段
    # 訂閱商品 R1 #########################
    api.quote.subscribe(
        api.Contracts.Futures.TXF.TXFR1,
        quote_type = sj.constant.QuoteType.BidAsk,
        version = sj.constant.QuoteVersion.v1)
    # 訂閱商品 R2 #########################
    api.quote.subscribe(
        api.Contracts.Futures.TXF.TXFR2,
        quote_type = sj.constant.QuoteType.BidAsk,
        version = sj.constant.QuoteVersion.v1)
    # 設定商品回傳
    api.quote.set_on_bidask_fop_v1_callback(bidask_callback_function_for_check)
    # timeout需設置 等待收到第一筆CALLBACK 
    start_time = time.time()
    while market_is_open_or_not != True:
        time.sleep(0.01)
        if time.time() - start_time >= timeout:
            break
    # 收到確定有開盤後取消訂閱#################
    api.quote.unsubscribe(
        api.Contracts.Futures.TXF.TXFR1,
        quote_type = sj.constant.QuoteType.BidAsk,
        version = sj.constant.QuoteVersion.v1)
    api.quote.unsubscribe(
        api.Contracts.Futures.TXF.TXFR2,
        quote_type = sj.constant.QuoteType.BidAsk,
        version = sj.constant.QuoteVersion.v1)
    # api.logout()
    return market_is_open_or_not
        
# 判斷TXO合約tick
def get_tick_size_TXO(price):
    if price <= 0:
        return None
    elif price < 10:
        return 0.1
    elif price < 50:
        return 0.5
    elif price < 500:
        return 1
    elif price < 1000:
        return 5
    elif price < 5000:
        return 10
    else:
        return None

# 判斷TXF合約tick
def get_tick_size_TXF(price):
    if price <= 0:
        return None
    elif price >= 1:
        return 1
    else:
        return None

# 下單合約進位_選擇權
def round_order_price_TXO(price):
    price = float(price)
    if price <= 0:
        return None
    elif price < 10:
        if round(price - price % 0.1 + 0.1, 2) == 10:
            return int(10)
        if round(price % 0.1, 2) < 0.05:
            return round(price - price % 0.1, 2)
        else:
            return round(price - price % 0.1 + 0.1, 2)
    elif price < 50:
        if price % 1 < 0.25:
            return int(price)
        elif price % 1 < 0.75:
            return int(price) + 0.5
        else:
            return int(price) + 1
    elif price < 500:
        if price % 1 < 0.5:
            return int(price)
        else:
            return int(price) + 1
    elif price < 1000:
        if price % 10 < 2.5:
            return int(price - price % 10)
        elif price % 10 < 7.5:
            return int(price - price % 10) + 5
        else:
            return int(price - price % 10) + 10
    elif price >= 1000:
        if price % 10 < 5:
            return int(price - price % 10)
        else:
            return int(price - price % 10) + 10
    else:
        return None

# 下單合約進位_期貨
def round_order_price_TXF(price):
    price = float(price)
    if price <= 0:
        return None
    elif price > 1:
        return int(round(price, 0))
    else:
        return None

def convert_yearmonth_code(yearmonth_code): # 輸入:'202307'輸出:('G3','S3')  輸入:"G3"or"S3" 輸出:"202307"
    # 定義對應字母月份代號的字典
    yearmonth_code_dict = {}
    year = 2020
    # 生成合約代號字典
    for i in range(120):
        # 第一個代號
        code1 = chr(ord('A') + i % 12) + str(year % 10)
        # 第二個代號（字母移動 12 個位置）
        code2 = chr(ord(f'{code1[0]}') + 12) + str(year % 10)
        # 以字典形式存入
        yearmonth_code_dict[f'{year}{(i % 12) + 1:02d}'] = (code1, code2)
        if i % 12 == 11:
            year += 1
    # 查找輸入對應的代號
    if yearmonth_code in yearmonth_code_dict:
        return yearmonth_code_dict[yearmonth_code]
    else:
        # 反向查找合約代號
        if yearmonth_code:
            for key, value in yearmonth_code_dict.items():
                if yearmonth_code in value:
                    return key
    return None

def add_one_month(date_str): # 簡單的月份增量函式，處理年月格式如 '202409' -> '202410'
    year = int(date_str[:4])
    month = int(date_str[4:])
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    return f"{year}{month:02}"

def is_today_month_contract_end(api): #今天是結算日嗎 函數 
    # 大台近月次月處理
    futures=list(api.Contracts.Futures.TXF)
    sorted_contracts = sorted([contract for contract in futures if not contract.code.startswith('TXFR')], key=lambda x: x.symbol[-6:])
    delivery_date = datetime.datetime.strptime(sorted_contracts[0].delivery_date, '%Y/%m/%d')
    update_date = datetime.datetime.strptime(sorted_contracts[0].update_date, '%Y/%m/%d')
    if delivery_date <= update_date : 
        return True, sorted_contracts[0]
    else:
        return False, sorted_contracts[0]

def clean_order_callback_feather(file_path):
    '''
    輸入 feather 檔後自動輸出 df
    '''
    df = pd.read_feather(file_path)

    if isinstance(df['message'].iloc[0], str):
        df['message'] = df['message'].apply(eval)

    def parse_row(row):
        msg = row['message']
        parsed = {
            'ts_local': row['timestamp'],
            'code': None,
            'delivery_month': None,
            'action': None,
            'quantity': None,
            'price': None,
            'market_type': None,
            'op_type': None,
            'op_code': None,
            'op_msg': None,
            'ts_exchange': None,
            'id': None,
            'order_quantity': None,
            'oc_type': None,
            'seqno': None,
            'ordno': None,
            'account_id': None,
            'combo': None
        }

        # 長格式
        if isinstance(msg, dict) and 'operation' in msg:
            parsed.update({
                'code': msg.get('contract', {}).get('code'),
                'delivery_month': msg.get('contract', {}).get('delivery_month'),
                'action': msg.get('order', {}).get('action'),
                'quantity': msg.get('order', {}).get('quantity'),
                'price': msg.get('order', {}).get('price'),
                'market_type': msg.get('order', {}).get('market_type'),
                'op_type': msg.get('operation', {}).get('op_type'),
                'op_code': msg.get('operation', {}).get('op_code'),
                'op_msg': msg.get('operation', {}).get('op_msg'),
                'ts_exchange': datetime.datetime.fromtimestamp(msg.get('status', {}).get('exchange_ts')) if msg.get('status', {}).get('exchange_ts') else None,
                'id': msg.get('status', {}).get('id'),
                'order_quantity': msg.get('status', {}).get('order_quantity'),
                'oc_type': msg.get('order', {}).get('oc_type'),
                'seqno': msg.get('order', {}).get('seqno'),
                'ordno': msg.get('order', {}).get('ordno'),
                'account_id': msg.get('order', {}).get('account', {}).get('account_id'),
                'combo': msg.get('order', {}).get('combo'),
            })
        # 短格式（成交訊息）
        elif isinstance(msg, dict) and 'ts' in msg and 'price' in msg and 'quantity' in msg:
            parsed.update({
                'code': msg.get('code'),
                'delivery_month': msg.get('delivery_month'),
                'action': msg.get('action'),
                'quantity': msg.get('quantity'),
                'price': msg.get('price'),
                'market_type': msg.get('market_type'),
                'ts_exchange': datetime.datetime.fromtimestamp(msg.get('ts')) if msg.get('ts') else None,
                'id': msg.get('trade_id') or msg.get('id'),
                'seqno': msg.get('seqno'),
                'ordno': msg.get('ordno'),
                'account_id': msg.get('account_id'),
                'combo': msg.get('combo'),
            })
        return parsed

    clean_df = pd.DataFrame(df.apply(parse_row, axis=1).tolist())
    clean_df = clean_df.sort_values('ts_local')
    return clean_df

# 時間常數
def get_market_hours_TXF():
    '''
    start_time_day_pre_opening = market_hours['start_time_day_pre_opening']  # 日盤試撮開盤時間
    start_time_night_pre_opening = market_hours['start_time_night_pre_opening']   # 夜盤試撮開盤時間
    start_time_day = market_hours['start_time_day']  # 日盤開盤時間
    end_time_day = market_hours['end_time_day']   # 日盤收盤時間
    start_time_night = market_hours['start_time_night']  # 夜盤開盤時間
    end_time_night = market_hours['end_time_night']     # 夜盤收盤時間
    end_time_day_contract_close = market_hours['end_time_day_contract_close']     # 日盤結算日收盤時間
    '''
    return {
        "start_time_day_pre_opening": datetime.time(8, 30, 0),
        "start_time_day": datetime.time(8, 45, 0),
        "end_time_day": datetime.time(13, 45, 0),
        "start_time_night_pre_opening": datetime.time(14, 50, 0),
        "start_time_night": datetime.time(15, 0, 0),
        "end_time_night": datetime.time(5, 0, 0),
        "end_time_day_contract_close": datetime.time(13, 30, 0),
    }

def map_letter_to_callput(letter):
    """判斷輸入字母是 call 還是 put 區間"""
    call_set = set("ABCDEFGHIJKL")   # 前12個小寫字母代表 call
    put_set = set("MNOPQRSTUVWX")    # 後12個小寫字母代表 put

    letter = letter.upper()

    if letter in call_set:
        return "call"
    elif letter in put_set:
        return "put"
    else:
        return "invalid input"

def map_letter_change_callput(letter):
    """將 call/put 對應字母互換，依照固定位置對應 A>M X>L"""
    call_list = list("ABCDEFGHIJKL")   # call 對應的字母
    put_list = list("MNOPQRSTUVWX")    # put 對應的字母

    letter = letter.upper()

    if letter in put_list:
        index = put_list.index(letter)
        return call_list[index]
    elif letter in call_list:
        index = call_list.index(letter)
        return put_list[index]
    else:
        return "INVALID INPUT"





'''
標準文件 轉成英文版
[ERROR] red \033[1;31m
[SUCCESS] green \033[1;32m
[OK] green \033[1;32m
[WARN] yellow \033[1;33m
[INFO] blue \033[1;34m
'''

'''
api.margin()
Margin(status=<FetchStatus.Fetched: 'Fetched'>, yesterday_balance=859379.0, today_balance=879359.0, deposit_withdrawal=0.0, fee=141.0, tax=179.0, initial_margin=364750.0, maintenance_margin=277500.0, margin_call=0.0, risk_indicator=247.0, royalty_revenue_expenditure=3200.0, equity=896309.0, equity_amount=891759.0, option_openbuy_market_value=0.0, option_opensell_market_value=4550.0, option_open_position=-1350.0, option_settle_profitloss=0.0, future_open_position=16949.99, today_future_open_position=15550.0, future_settle_profitloss=17100.0, available_margin=510809.0, plus_margin=0.0, plus_margin_indicator=0.0, security_collateral_amount=0.0, order_margin_premium=0.0, collateral_amount=0.0)
api.list_positions()
[FuturePosition(id=0, code='MXFF5', direction=<Action.Buy: 'Buy'>, quantity=3, price=21624.0, last_price=21678.0, pnl=8100.0),
 FuturePosition(id=1, code='TX222000F5', direction=<Action.Sell: 'Sell'>, quantity=1, price=64.0, last_price=74.0, pnl=-500.0)]

 ✅ [19] 訂閱合約: code='TX120800F5' symbol='TX120250620800C' name='臺指選擇權06W1月20800C' category='TX1' delivery_month='202506' delivery_date='2025/06/04' strike_price=20800.0 option_right=<OptionRight.Call: 'C'> underlying_kind='I' unit=1 limit_up=2390.0 limit_down=0.1 reference=296.0 update_date='2025/06/03' 成功

'''


# 2023/05/17  13:31 R1R2 依然照舊 5月依然可以Quote 只是沒有回傳
# 其實可以try 前一個月份合約 本月份合約 下月份合約 TRY到有成功quote為止
# 判斷開盤的條件 1.日期是否結算日 是否第三個禮拜 是否有颱風假 是否有年假 2.是否合約有quote訂閱回傳
# 13:30需自動切換進月台指 換月
# 3.是否為開盤時段4.是否snapshot可訂閱該合約
# 2023/05/17 13:45 時間一到API可能跳開，但依然可以quote 只是沒有回傳的最新報價

'''
行情 :credit_enquire, short_stock_sources, snapshots, ticks, kbars
以上查詢總次數5秒上限500次
帳務 :list_profit_loss_detail,account_balance, list_settlements, list_profit_loss, list_positions
以上查詢總次數5秒上限25次
委託 :place_order, update_status, update_qty, update_price, cancel_order
以上查詢總次數10秒上限500次
連線 :同一永豐金證券person_id,僅可使用最多5個連線。 注意: api.login()即建立一個連線
登入 :api.login()一天上限1000次
訂閱 :api.quote.subscribe()最多250個合約
2023/9/15
行情 :credit_enquire, short_stock_sources, snapshots, ticks, kbars
以上查詢總次數 5 秒上限 50 次
盤中查詢 ticks 次數不得超過 10 次
盤中查詢 kbars 次數不得超過 270 次
帳務 :list_profit_loss_detail,account_balance, list_settlements, list_profit_loss, list_positions
以上查詢總次數 5 秒上限 25 次
委託 :place_order, update_status, update_qty, update_price, cancel_order
以上查詢總次數 10 秒上限 500 次
連線 :同一永豐金證券person_id,僅可使用最多5個連線。 注意: api.login()即建立一個連線
登入 :api.login()一天上限1000次
'''
'''
list_profit_loss_detail,list_profit_loss已實現損益
account_balance,list_settlements 股票帳戶餘額
list_positions 當前倉位
'''
'''
台指試搓時間  8:30-8:45 43、44無法刪單
夜盤試搓時間 14:50-15:00 58 59無法刪單

訂閱合約回傳
# Code
# datetime
# open
# underlying_price
# bid_side_total_vol
# ask_side_total_vol
# avg_price
# close
# high
# low
# amount
# total_amount
# volume
# total_volume
# tick_type
# chg_type
# price_chg
# pct_chg
# simtrade

'''
'''
非交易時間或是取得錯誤都會回報 偏伺服器端錯誤
'NoneType' object has no attribute 'code'
找不到合約 偏用戶端錯誤
'NoneType' object is not subcriptable
'''
