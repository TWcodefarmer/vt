# === 標準庫 Standard Library ===
import os
import glob
import datetime
import time
import re
import zipfile
import logging

# === 第三方套件 Third-party Packages ===
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from bs4 import BeautifulSoup
import requests

# === 日誌 logging 設定 Logging Settings ===
logging.basicConfig(
    filename='__Update_TFE_Database.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def data_cleansing_TFX_futures_OriginalTickFiles(filename, contract):
    '''
    input filename: 原始期交所期貨RPT or CSV檔
    contract: 預期處理合約名稱
    output: 清洗後df
    '''
    # 讀取CSV檔案並指定編碼為Big5
    df = pd.read_csv(filename, encoding='Big5',low_memory=False)
    # 指定英文欄名
    eng_columnames = ['date', 'contract', 'duedmonth', 'timestamp', 'price', 'volume', '1', '2', '3']
    # 替換資料框的欄名為英文欄名
    df.columns = eng_columnames
    # 移除contract和duedmonth字串欄位的前後空白字元
    df['contract'] = df['contract'].str.strip().astype('string').astype('category')
    df['duedmonth'] = df['duedmonth'].str.strip().astype('string').astype('category')
    # 選擇contract為"TX"
    df = df.query('contract == @contract').iloc[:, 0:6]
    # 將date和timestamp欄位的資料型別轉換為字串
    df['date'] = df['date'].astype('int64').astype('string')
    df['timestamp'] = df['timestamp'].astype('int64').astype('string').str.zfill(6)
    # 新增datetime欄位，合併date和timestamp欄位
    df['datetime'] = df['date'] + df['timestamp']
    # 將datetime欄位轉換為日期時間型別
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
    # 根據datetime欄位進行排序
    df = df.sort_values('datetime')
    # 計算cap欄位，成交量為買賣合併
    df['cap'] = df['price'] * df['volume']
    # 整理欲輸出格式
    df.drop(['date', 'timestamp'], axis=1,inplace=True)
    df.insert(0, 'datetime', df.pop('datetime'))
    # 重設index 加速處理速度
    df.reset_index(drop=True, inplace=True)
    return df

def data_cleansing_TFX_options_OriginalTickFiles(filename, contract):
    '''
    input filename: 原始期交所選擇權RPT or CSV檔
    contract: 預期處理合約名稱
    output: 清洗後df
    '''
    # 讀取CSV檔案並指定編碼為Big5
    df = pd.read_csv(filename, encoding='Big5',low_memory=False)
    # 指定英文欄名
    eng_columnames = ['date', 'contract', 'strike', 'duedmonth', 'callput', 'timestamp', 'price', 'volume', '1']
    # 替換資料框的欄名為英文欄名
    df.columns = eng_columnames
    # 移除contract和duedmonth字串欄位的前後空白字元
    df['contract'] = df['contract'].str.strip().astype('string').astype('category')
    df['duedmonth'] = df['duedmonth'].str.strip().astype('string').astype('category')
    df['callput'] = df['callput'].str.strip().astype('string').astype('category')
    # # 選擇contract為"TX"且duedmonth為"202307"的資料並選取全部欄位
    df = df.query('contract == @contract').iloc[:, 0:8]
    # # 將date和timestamp欄位的資料型別轉換為字串
    df['date'] = df['date'].astype('int64').astype('string')
    df['timestamp'] = df['timestamp'].astype('int64').astype('string').str.zfill(6)
    df['strike'] = df['strike'].astype('int64').astype('string').astype('category')
    # 新增datetime欄位，合併date和timestamp欄位
    df['datetime'] = df['date'] + df['timestamp']
    # 將datetime欄位轉換為日期時間型別
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
    # # 根據datetime欄位進行排序
    df = df.sort_values('datetime')
    # # 計算cap欄位，成交量為買賣合併
    df['cap'] = df['price'] * df['volume']
    # # 整理欲輸出格式
    df.drop(['date', 'timestamp'], axis=1,inplace=True)
    df.insert(0, 'datetime', df.pop('datetime'))
    # 重設index 加速處理速度
    df.reset_index(drop=True, inplace=True)
    return df

def future_crawler(CSVorRPT, wait_time, target_folder='TFE_tick_rpt_fut'):
    # 確保目標資料夾存在
    os.makedirs(target_folder, exist_ok=True)

    # 判斷要下載的檔案類型
    filetype = 'CSV' if CSVorRPT.lower() == 'csv' else ''

    # 期交所近30日網址
    target = 'https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData'
    response = requests.get(target)
    sp = BeautifulSoup(response.text, 'html.parser')
    targetsouplist = sp.find_all('input', class_='btn_orange')

    # 找到所有可下載檔案的日期
    filedates = []
    for i in targetsouplist:
        if 'https://www.taifex.com.tw/file/taifex/Dailydownload/Dailydownload/Daily' in str(i):
            y_m_d = (str(i).split('_')[2:4] + str(i).split('_')[4:5][0].split('.'))[0:3]
            ymd = '_'.join(y_m_d)
            if ymd not in filedates:
                filedates.append(ymd)

    # 找到所有發布時間
    realeasetimelist = []
    targetsouplist = sp.find_all('td', align='center')
    pattern = r'\d{4}/\d{2}/\d{2} (AM|PM) \d{2}:\d{2}:\d{2}'
    for i in targetsouplist:
        match = re.search(pattern, str(i))
        if match:
            i = str(i).split('>')[1].split('<')[0]
            realeasetimelist.append(i)

    df1 = None
    if len(filedates) == len(realeasetimelist):
        df1 = pd.DataFrame({'發布時間': realeasetimelist, '檔案日期': filedates})

    for i, d in enumerate(filedates):
        realeasedate = datetime.datetime.strptime(realeasetimelist[i], '%Y/%m/%d %p %I:%M:%S').strftime('%Y_%m_%d')
        releasetime = datetime.datetime.strptime(realeasetimelist[i], '%Y/%m/%d %p %I:%M:%S').time()

        if d == realeasedate and releasetime >= datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0).time():
            downloadlink = f'https://www.taifex.com.tw/file/taifex/Dailydownload/Dailydownload{filetype}/Daily_{d}.zip'
            filename = downloadlink.split('/')[-1]
            file_path = os.path.join(target_folder, filename)
            rpt_path = os.path.join(target_folder, filename.replace('.zip', '.rpt'))

            if os.path.exists(file_path) or os.path.exists(rpt_path):
                print('此期貨已存在，跳過不下載:', filename)
            else:
                with urllib.request.urlopen(downloadlink) as response:
                    zipcontent = response.read()
                    with open(file_path, 'wb') as f:
                        f.write(zipcontent)
                    print(f'下載成功: {filename}')
                    for j in range(wait_time):
                        print(f'休息{j + 1}秒')
                        time.sleep(1)
        else:
            print(f'這天 {d} 檔案目前只有半天夜盤，跳過')
    print('下載任務執行完畢')
    return df1

def option_crawler(CSVorRPT, wait_time, target_folder='TFE_tick_rpt_opt'):
    # 確保資料夾存在
    os.makedirs(target_folder, exist_ok=True)

    # 決定檔案類型
    filetype = 'CSV' if CSVorRPT.lower() == 'csv' else ''

    # 選擇權近30日資料網址
    target = 'https://www.taifex.com.tw/cht/3/dlOptPrevious30DaysSalesData'
    response = requests.get(target)
    sp = BeautifulSoup(response.text, 'html.parser')
    targetsouplist = sp.find_all('input', class_='btn_orange')

    # 抓日期
    filedates = []
    for i in targetsouplist:
        if 'https://www.taifex.com.tw/file/taifex/Dailydownload/OptionsDailydownload/OptionsDaily' in str(i):
            y_m_d = (str(i).split('_')[2:4] + str(i).split('_')[4:5][0].split('.'))[0:3]
            ymd = '_'.join(y_m_d)
            if ymd not in filedates:
                filedates.append(ymd)

    # 抓發布時間
    realeasetimelist = []
    targetsouplist = sp.find_all('td', align='center')
    pattern = r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}'
    for i in targetsouplist:
        match = re.search(pattern, str(i))
        if match:
            i = str(i).split('>')[1].split('<')[0]
            realeasetimelist.append(i)

    df2 = None
    if len(filedates) == len(realeasetimelist):
        df2 = pd.DataFrame({'發布時間': realeasetimelist, '檔案日期': filedates})

    # 開始下載
    for i, d in enumerate(filedates):
        realeasedate = datetime.datetime.strptime(realeasetimelist[i], '%Y/%m/%d %H:%M:%S').strftime('%Y_%m_%d')
        releasetime = datetime.datetime.strptime(realeasetimelist[i], '%Y/%m/%d %H:%M:%S').time()

        if d == realeasedate and releasetime >= datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0).time():
            downloadlink = f'https://www.taifex.com.tw/file/taifex/Dailydownload/OptionsDailydownload{filetype}/OptionsDaily_{d}.zip'
            filename = downloadlink.split('/')[-1]
            file_path = os.path.join(target_folder, filename)
            rpt_path = os.path.join(target_folder, filename.replace('.zip', '.rpt'))

            if os.path.exists(file_path) or os.path.exists(rpt_path):
                print('此選擇權檔案已存在，跳過不下載:', filename)
            else:
                with urllib.request.urlopen(downloadlink) as response:
                    zipcontent = response.read()
                    with open(file_path, 'wb') as f:
                        f.write(zipcontent)
                    print(f'下載成功: {filename}')
                    for j in range(wait_time):
                        print(f'休息{j + 1}秒')
                        time.sleep(1)
        else:
            print(f'這天 {d} 檔案只有半天夜盤，跳過')
    print('選擇權資料下載完畢')
    return df2

def extract_zip_files(target_folder):
    # 確保目標資料夾存在
    os.makedirs(target_folder, exist_ok=True)

    existing_rpts = set()
    for filename in os.listdir(target_folder):
        if filename.endswith('.rpt'):
            existing_rpts.add(filename)

    for filename in os.listdir(target_folder):
        if filename.endswith('.zip') and filename.replace('.zip', '.rpt') not in existing_rpts:
            with zipfile.ZipFile(os.path.join(target_folder, filename), 'r') as zip_ref:
                zip_ref.extractall(target_folder)
            print(f'{filename} 已成功解壓縮。')

def get_tick_data_from_TFE():
    """
    從期交所 (TFE) 下載並解壓期貨、選擇權逐筆資料 (tick data)。
    Download and extract futures & options tick data from TFE.

    步驟：
    1. 下載期貨資料並解壓。
    2. 下載選擇權資料並解壓。
    """
    # 設定期貨 tick 資料儲存的資料夾名稱
    foldername = 'TFE_tick_rpt_fut'
    # 下載期貨 tick 資料，等待時間設為 3 秒
    future_crawler('rpt', wait_time=1, target_folder=foldername)
    # 解壓下載的 zip 檔案至指定資料夾
    extract_zip_files(target_folder=foldername)

    # 設定選擇權 tick 資料儲存的資料夾名稱
    foldername = 'TFE_tick_rpt_opt'
    # 下載選擇權 tick 資料，等待時間設為 3 秒
    option_crawler('rpt', wait_time=1, target_folder=foldername)
    # 解壓下載的 zip 檔案至指定資料夾
    extract_zip_files(target_folder=foldername)

def combine_rpts_to_single_csv(folder_name):
    """
    將指定資料夾內所有 .rpt 檔案合併成一個未清洗的大 CSV 檔案，並儲存於當前工作目錄。

    參數:
        folder_name (str): 包含 .rpt 檔案的資料夾名稱。

    回傳:
        str: 合併後儲存的 CSV 檔案名稱（含時間戳記）。
    """
    # 建立資料夾的完整路徑
    folder_path = os.path.join(os.getcwd(), folder_name)

    # 尋找該資料夾內所有 .rpt 檔案（以 Big5 編碼儲存的 CSV 格式）
    rpt_files = glob.glob(os.path.join(folder_path, '*.rpt'))

    # 將所有 .rpt 檔案讀入為 DataFrame 並存入清單
    dataframes = [pd.read_csv(file, encoding='Big5', low_memory=False) for file in rpt_files]

    # 將所有資料合併為一個大 DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # 產生儲存用檔案名稱（帶有時間戳記）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'{folder_name}_uncleaned_{timestamp}.csv'

    # 將合併後資料儲存為 Big5 編碼的 CSV 檔案
    combined_df.to_csv(output_filename, index=False, encoding='Big5')

    return output_filename

def update_futures_feather_db(
    feather_db='near_fut_cleaned.feather', 
    rpt_folder='TFE_tick_rpt_fut',
    contract_code='TX'
):
    """自動整合並更新台指期 feather 資料庫
    
    參數:
        feather_db (str): feather 檔案名稱
        rpt_folder (str): rpt 檔案所在資料夾
        contract_code (str): 合約代碼（例如 TX）
    """
    try:
        if os.path.exists(feather_db):
            df_0 = pd.read_feather(feather_db)
            time_first = df_0.iloc[0, 0]   # 最舊的日期 first date
            time_last = df_0.iloc[-1, 0]   # 最新的日期 last date
            print('檔案名稱', feather_db, '起始日期:', time_first, '最新日期:', time_last)
            folder_path = os.path.join(os.getcwd(), rpt_folder)
            rpt_files = glob.glob(os.path.join(folder_path, '*.rpt'))
            for file_path in rpt_files:
                # 擷取 rpt 檔名中的日期
                try:
                    time_1 = datetime.datetime.strptime(
                        (''.join(file_path.split('Daily_')[1:2]).split('.')[0]), '%Y_%m_%d')
                except Exception as e:
                    print(f"檔名解析失敗: {file_path}, 錯誤: {e}")
                    continue
                if time_last < time_1:
                    print('新增資料庫日期:', time_1)
                    newdf = data_cleansing_TFX_futures_OriginalTickFiles(file_path, contract_code)
                    df_0 = pd.concat([df_0, newdf], ignore_index=True)
                    df_0['contract'] = df_0['contract'].astype('category')
                    df_0['duedmonth'] = df_0['duedmonth'].astype('category')
            df_0.to_feather(feather_db)
        else:
            print(f"檔案 '{feather_db}' 不存在，製造新資料庫")
            # 假設 combine_rpts_to_single_csv 輸出整合過的檔名
            filename = combine_rpts_to_single_csv(rpt_folder)
            # 清洗原始大CSV 並存成feather
            data_cleansing_TFX_futures_OriginalTickFiles(filename, contract_code).to_feather(feather_db)
        
        # 檢查日期是否遞增 (is_monotonic_increasing 英文單字: 單調遞增)
        check_date_increasing = pd.read_feather(feather_db)['datetime'].is_monotonic_increasing
        logging.info(f' : {check_date_increasing} : 台指日期是否正確遞增')
        return check_date_increasing
    except Exception as ex:
        logging.error(f"自動整合期貨資料庫發生異常: {ex}")
        return False

def update_options_feather_db(
    feather_db='near_opt_cleaned.feather',
    rpt_folder='TFE_tick_rpt_opt',
    contract_code='TXO'
):
    """
    自動整合並更新台指選擇權 feather 資料庫

    參數:
        feather_db (str): feather 檔案名稱
        rpt_folder (str): rpt 檔案資料夾名稱
        contract_code (str): 合約代碼（如 'TXO'）
    回傳:
        bool: 日期是否單調遞增（is_monotonic_increasing 英文單字：單調遞增）
    """
    try:
        if os.path.exists(feather_db):
            df_0 = pd.read_feather(feather_db)
            time_first = df_0.iloc[0, 0]   # 最舊的日期 first date
            time_last = df_0.iloc[-1, 0]   # 最新的日期 last date
            print('檔案名稱', feather_db, '起始日期:', time_first, '最新日期:', time_last)
            folder_path = os.path.join(os.getcwd(), rpt_folder)
            rpt_files = glob.glob(os.path.join(folder_path, '*.rpt'))
            for file_path in rpt_files:
                try:
                    time_1 = datetime.datetime.strptime(
                        (''.join(file_path.split('Daily_')[1:2]).split('.')[0]), '%Y_%m_%d')
                except Exception as e:
                    print(f"檔名解析失敗: {file_path}, 錯誤: {e}")
                    continue
                if time_last < time_1:
                    print('新增資料庫日期:', time_1)
                    newdf = data_cleansing_TFX_options_OriginalTickFiles(file_path, contract_code)
                    df_0 = pd.concat([df_0, newdf], ignore_index=True)
                    # 各欄位型別轉換
                    df_0['contract'] = df_0['contract'].astype('category')
                    df_0['duedmonth'] = df_0['duedmonth'].astype('category')
                    df_0['strike'] = df_0['strike'].astype('category')
                    df_0['callput'] = df_0['callput'].astype('category')
            df_0.to_feather(feather_db)
        else:
            print(f"檔案 '{feather_db}' 不存在，製造新資料庫")
            filename = combine_rpts_to_single_csv(rpt_folder)
            data_cleansing_TFX_options_OriginalTickFiles(filename, contract_code).to_feather(feather_db)

        # 檢查日期是否單調遞增（is_monotonic_increasing）
        check_date_increasing = pd.read_feather(feather_db)['datetime'].is_monotonic_increasing
        logging.info(f' : {check_date_increasing} : 台選日期是否正確遞增')
        return check_date_increasing
    except Exception as ex:
        logging.error(f"自動整合台指選擇權資料庫發生異常: {ex}")
        return False

def process_futures_minute_data(
    feather_path='near_fut_cleaned.feather',
    output_csv='RESULT_contract_end_times.csv',
    output_feather='RESULT_fut.feather',
    start_year=2011,
    end_year=None,
    pre_min=30
):
    """
    期貨資料分K與結算日製作 (Futures minute resampling & contract settlement day)

    參數說明:
        feather_path: 輸入 feather 檔案路徑
        output_csv: 輸出每月結算日 CSV
        output_feather: 輸出連續合約 feather
        start_year: 處理起始年
        end_year: 處理結束年 (預設今年)
        pre_min: 提前多少分鐘視為結束時間 (預設30)
    """

    # 預設到今年
    if end_year is None:
        end_year = datetime.datetime.now().year

    df = pd.read_feather(feather_path)
    df.set_index('datetime', inplace=True)

    # 分鐘級別聚合（resample）
    resampled_df = df.groupby(['duedmonth'], observed=True).resample(
        'min', label='right', closed='right'
    ).agg({'volume': 'sum', 'cap': 'sum'}).reset_index()

    # 重新計算價格欄位
    resampled_df['price'] = resampled_df['cap'] / resampled_df['volume']

    # 初始化 (initialize) 變數
    target_df = pd.DataFrame()
    latest_time = None
    end_times = []

    for i in range(start_year, end_year+2):
        for j in range(1, 13):
            duedmonth = str(i) + str(j).zfill(2)
            current_df = resampled_df.query('duedmonth == @duedmonth')
            if current_df.empty:
                continue
            # 結束時間點 (提前N分鐘)
            datetime_ = current_df.iloc[-1].datetime - pd.Timedelta(minutes=pre_min)
            datetime_plus_1 = datetime_ + pd.Timedelta(minutes=1)
            matched_rows = current_df[current_df['datetime'] == datetime_]
            if matched_rows.empty:
                continue
            index_start = current_df.iloc[0].name
            index_end = matched_rows.index[0]
            end_price = matched_rows.price.iloc[0]
            end_times.append({'duedmonth': duedmonth, 'end_time': datetime_})
            new_target = current_df.loc[index_start:index_end]
            if latest_time:
                new_target = new_target[new_target['datetime'] >= latest_time]
            latest_time = datetime_plus_1
            target_df = pd.concat([target_df, new_target], ignore_index=True)

    # 結算日處理
    end_times_df = pd.DataFrame(end_times)
    end_times_df['time_only'] = pd.to_datetime(end_times_df['end_time']).dt.time
    # 僅保留 13:00:00 結算的合約
    end_times_df = end_times_df[end_times_df['time_only'] == datetime.time(13, 0, 0)]
    end_times_df = end_times_df.drop(columns=['time_only'])
    # 只保留當月以前的資料
    current_ym = int(datetime.datetime.now().strftime("%Y%m"))
    end_times_df = end_times_df[end_times_df['duedmonth'].astype(int) <= current_ym]

    # 輸出
    end_times_df.to_csv(output_csv, index=False)
    print(f"✅ 每月合約結算日已儲存 {output_csv}")
    print('最近一次合約結算月份:', end_times_df.tail(1).duedmonth.values[0])

    target_df = target_df.dropna(subset=['price']).reset_index(drop=True)
    target_df.to_feather(output_feather)
    print(f"✅ 台指分K已清理且合併為連續合約 {output_feather}")

    return end_times_df, target_df

def calculate_contract_price_gap(
    feather_path='near_fut_cleaned.feather',
    contract_end_csv='RESULT_contract_end_times.csv',
    output_csv='RESULT_contract_price_gap.csv',
    max_attempts=20
):
    """
    計算合約價差（spread/gap）並儲存為 CSV

    參數說明:
        feather_path: 近月期貨資料 feather 路徑
        contract_end_csv: 合約結束日 csv 路徑
        output_csv: 結果儲存路徑
        max_attempts: 每個結束日最多往前幾分鐘找（預設 20）
    """
    near_fut_df = pd.read_feather(feather_path)
    # 過濾掉 'duedmonth' 含 '/' 的資料（格式異常）
    near_fut_df = near_fut_df[~near_fut_df['duedmonth'].astype(str).str.contains('/')]
    contract_dates_df = pd.read_csv(contract_end_csv)
    results = []

    for idx, row in contract_dates_df.iterrows():
        temp_datetime = row.iloc[1]  # 結算時間（datetime）
        found_data = False

        for _ in range(max_attempts):
            temp_df = near_fut_df[near_fut_df['datetime'] == temp_datetime]
            if not temp_df.empty:
                temp_df = temp_df.sort_values('duedmonth')
                contracts = temp_df.duedmonth.unique()
                if len(contracts) > 1:
                    contract_now = contracts[0]
                    contract_next = contracts[1]
                    # 驗證第一個合約必須為當月
                    current_month = pd.to_datetime(temp_datetime).strftime('%Y%m')
                    if contract_now != current_month:
                        print(f"❌ ERROR: 合約 {contract_now} 不符合日期 {temp_datetime} 的當月 {current_month}")
                        temp_datetime = pd.to_datetime(temp_datetime) - pd.Timedelta(minutes=1)
                        continue
                    contract_now_price = (
                        temp_df.query('duedmonth == @contract_now').cap.sum() /
                        temp_df.query('duedmonth == @contract_now').volume.sum()
                    )
                    contract_next_price = (
                        temp_df.query('duedmonth == @contract_next').cap.sum() /
                        temp_df.query('duedmonth == @contract_next').volume.sum()
                    )
                    gap = contract_next_price - contract_now_price
                    results.append({
                        'datetime': temp_datetime,
                        'contract_now': contract_now,
                        'contract_next': contract_next,
                        'contract_now_price': contract_now_price,
                        'contract_next_price': contract_next_price,
                        'gap': gap
                    })
                    found_data = True
                    break
            # 時間往前推移一分鐘
            temp_datetime = pd.to_datetime(temp_datetime) - pd.Timedelta(minutes=1)
        if not found_data:
            print(f"⚠️ 無法找到 {row.iloc[1]} 附近的交易資料，跳過該時間點。")

    # 轉成 DataFrame 並累加 gap
    results_df = pd.DataFrame(results)
    results_df['gap_cumsum'] = results_df.gap.cumsum()
    results_df.to_csv(output_csv, index=False)
    print(f"✅ 資料庫的合約價差已儲存: '{output_csv}'")
    # 合約名稱檢查
    contracts = sorted(results_df["contract_now"].tolist())
    print("合約名稱:", contracts)

    # 可選: 合約連續性檢查
    # is_continuous = all((int(contracts[i + 1]) - int(contracts[i])) in {1, 11} for i in range(len(contracts) - 1))
    # if is_continuous:
    #     print("✅無須理會上述 ERROR 合約連續無缺漏 ")
    # else:
    #     print("🔍 合約連續性檢查: ❌ 合約不連續，存在缺漏 檢查上述 ERROR")
    return results_df

def adjust_continuous_contract_gap(
    fut_feather='RESULT_fut.feather',
    price_gap_csv='RESULT_contract_price_gap.csv',
    output_feather='RESULT_gap_fixed.feather'
):
    """
    處理連續合約換月價差調整 (adjust continuous contract price gap)

    參數說明:
        fut_feather: 讀取的連續合約feather檔案
        price_gap_csv: 合約價差結果csv
        output_feather: 調整後儲存的feather檔案
    """

    # 讀取連續合約分K資料
    target_df = pd.read_feather(fut_feather)
    # 讀取價差資料
    df = pd.read_csv(price_gap_csv)
    # 準備gap字典
    gap_df = df[['contract_next', 'gap_cumsum']].copy()
    gap_df['contract_next'] = gap_df['contract_next'].astype(str)
    gap_dict = dict(zip(gap_df['contract_next'], gap_df['gap_cumsum']))
    # 型態統一
    target_df['duedmonth'] = target_df['duedmonth'].astype(str)
    # 調整價格 adjusted_price
    target_df['price'] = target_df.apply(
        lambda row: row['price'] - gap_dict.get(row['duedmonth'], 0),
        axis=1
    )
    # cap重新計算
    target_df['cap'] = (target_df['volume'] * target_df['price']) // 1
    # 儲存
    target_df.to_feather(output_feather)
    print(f"✅ 調整後的價格已儲存至 '{output_feather}'")
    return target_df

def plot_price_by_duedmonth(df, start, end): # 月合約型態確認 可觀察合約是否聚合錯誤
    """
    繪製指定區間內的價格圖，根據 duedmonth 分組，並設置黑色背景。

    參數:
    df : DataFrame
        包含 'duedmonth' 和 'price' 欄位的資料集。
    start : int
        起始索引。
    end : int
        結束索引。
    """
    # 篩選指定區間的資料
    filtered_df = df.iloc[start:end]

    # 設置黑底樣式
    plt.style.use('dark_background')

    # 使用索引作為 x 軸來繪製圖形
    plt.figure(figsize=(10, 6))
    for duedmonth, group in filtered_df.groupby(['duedmonth'], observed=True):
        plt.plot(group.index, group['price'], label=f"duedmonth: {duedmonth}")

    plt.xlabel('Index', color='white')
    plt.ylabel('Price', color='white')
    plt.title(f'Price vs Index by Duedmonth (Rows {start} to {end})', color='white')
    plt.legend(facecolor='black', edgecolor='white')
    plt.tight_layout()
    plt.show()


get_tick_data_from_TFE()

check = update_futures_feather_db()
print("台指日期是否正確遞增:", check)

check = update_options_feather_db()
print("台選日期是否正確遞增:", check)

end_times_df, target_df = process_futures_minute_data(pre_min=30) # 可調提前分鐘數
calculate_contract_price_gap(max_attempts=20) # 可自訂搜尋分鐘數
adjusted_df = adjust_continuous_contract_gap()

# debug 工具 觀看合約價格連續性
# plot_price_by_duedmonth(target_df,0,-1)
# plot_price_by_duedmonth(adjusted_df,0,-1)

def recycle_temp_files():
    # Remove .rpt files in ./TFE_tick_rpt_fut
    for rpt_file in glob.glob('./TFE_tick_rpt_fut/*.rpt'):
        try:
            os.remove(rpt_file)
            print(f"\033[1;32m[OK]\033[0m Deleted: {rpt_file}")
        except Exception as e:
            print(f"\033[1;31m[ERROR]\033[0m Failed to delete: {rpt_file}, reason: {e}")

    # Remove .rpt files in ./TFE_tick_rpt_opt
    for rpt_file in glob.glob('./TFE_tick_rpt_opt/*.rpt'):
        try:
            os.remove(rpt_file)
            print(f"\033[1;32m[OK]\033[0m Deleted: {rpt_file}")
        except Exception as e:
            print(f"\033[1;31m[ERROR]\033[0m Failed to delete: {rpt_file}, reason: {e}")

    # Remove TFE_tick_rpt_fut_uncleaned_*.csv in the main directory
    for csv_file in glob.glob('./*TFE_tick_rpt_fut_uncleaned_*.csv'):
        try:
            os.remove(csv_file)
            print(f"\033[1;32m[OK]\033[0m Deleted: {csv_file}")
        except Exception as e:
            print(f"\033[1;31m[ERROR]\033[0m Failed to delete: {csv_file}, reason: {e}")

    # Remove TFE_tick_rpt_opt_uncleaned_*.csv in the main directory
    for csv_file in glob.glob('./*TFE_tick_rpt_opt_uncleaned_*.csv'):
        try:
            os.remove(csv_file)
            print(f"\033[1;32m[OK]\033[0m Deleted: {csv_file}")
        except Exception as e:
            print(f"\033[1;31m[ERROR]\033[0m Failed to delete: {csv_file}, reason: {e}")

# 用法
recycle_temp_files()
