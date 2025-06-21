import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import ast 

class CSVSorterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CSV 閱讀與排序")
        self.master.configure(bg="#121212")  # 更暗的視窗底色

        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1c1c1c",
            fieldbackground="#1c1c1c",
            foreground="#b3b3b3",
            rowheight=30,
            font=("Arial", 14, "bold")
        )
        style.configure(
            "Treeview.Heading",
            background="#333333",
            foreground="#d9d9d9",
            relief="flat",
            font=("Arial", 12, "bold")
        )
        style.map("Treeview.Heading", background=[("active", "#444444")])

        # 讀取 CSV 按鈕
        self.load_csv_button = tk.Button(
            master,
            text="讀取 CSV",
            command=self.load_csv, # 這裡的 command 仍然是 load_csv
            bg="#333333",
            fg="#d9d9d9",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"),
            padx=10, pady=5
        )
        self.load_csv_button.pack(pady=10)

        # 下拉選單標籤
        self.sort_label = tk.Label(
            master, text="選擇排序欄位：", bg="#121212", fg="#d9d9d9", font=("Arial", 12)
        )
        self.sort_label.pack()

        # 第一排序欄位下拉選單
        self.sort_column = ttk.Combobox(master, state="readonly", font=("Arial", 12))
        self.sort_column.pack(pady=10)

        # 第二排序欄位下拉選單
        self.secondary_sort_label = tk.Label(
            master, text="選擇第二排序欄位：", bg="#121212", fg="#d9d9d9", font=("Arial", 12)
        )
        self.secondary_sort_label.pack()
        self.secondary_sort_column = ttk.Combobox(master, state="readonly", font=("Arial", 12))
        self.secondary_sort_column.pack(pady=10)

        # 遞增 / 遞減選項
        self.order_var = tk.StringVar(value="遞增")
        self.radio_frame = tk.Frame(master, bg="#121212")
        self.radio_frame.pack(pady=5)
        self.radio_asc = tk.Radiobutton(
            self.radio_frame, text="遞增", variable=self.order_var, value="遞增",
            bg="#121212", fg="#d9d9d9", selectcolor="#1c1c1c", font=("Arial", 12)
        )
        self.radio_desc = tk.Radiobutton(
            self.radio_frame, text="遞減", variable=self.order_var, value="遞減",
            bg="#121212", fg="#d9d9d9", selectcolor="#1c1c1c", font=("Arial", 12)
        )
        self.radio_asc.pack(side="left", padx=10)
        self.radio_desc.pack(side="left", padx=10)

        # 手動排序按鈕
        self.sort_button = tk.Button(
            master,
            text="排序",
            command=self.sort_data,
            bg="#333333",
            fg="#d9d9d9",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"),
            padx=10, pady=5
        )
        self.sort_button.pack(pady=10)

        # Treeview 顯示區
        self.tree = ttk.Treeview(master, show="headings", style="Treeview", selectmode="extended")
        self.tree.pack(expand=True, fill="both", padx=10, pady=10)

        # 勾選與標籤按鈕
        self.tag_button = tk.Button(
            master,
            text="標記選取列",
            command=self.tag_selected,
            bg="#333333",
            fg="#d9d9d9",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"),
            padx=10, pady=5
        )
        self.tag_button.pack(pady=10)

        # 綁定快捷鍵標記功能
        self.master.bind("<Control-t>", self.tag_selected_shortcut)
        self.tree.bind("<space>", self.toggle_selection)

        # 綁定滑鼠點擊事件 (在標頭上點擊)
        self.tree.bind("<Button-1>", self.on_click_heading)
        self.tree.bind("<Double-1>", self.copy_to_clipboard)

        self.csv_df = None
        self.current_sort_col = None
        self.is_ascending = True

        # 這裡改成存「DataFrame 的 index」，而非 TreeView 的 index
        self.tags = set()

        # 新增「更新 CSV」按鈕
        self.update_csv_button = tk.Button(
            master,
            text="更新當前 CSV",
            command=self.update_csv,
            bg="#333333",
            fg="#d9d9d9",
            activebackground="#444444",
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"),
            padx=10, pady=5
        )
        self.update_csv_button.pack(pady=10)

        # 只載入一次預設 CSV 檔案
        default_csv_path = "optimization_results___ParamOptimizer.csv"  # 修改為實際檔案路徑
        self.current_csv_path = default_csv_path
        
        # 初始載入時，直接呼叫 load_csv，load_csv 內部會處理排序
        self.load_csv(default_csv_path)

    def update_csv(self):
        if self.csv_df is None or self.current_csv_path is None:
            return
        self.csv_df.to_csv(self.current_csv_path, index=False)
        print("CSV 已更新！")

    def load_csv(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="選擇 CSV 檔案",
                filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")]
            )
            if not file_path:
                return
        self.current_csv_path = file_path
        self.csv_df = pd.read_csv(file_path)

        # 欄位名稱映射表
        columns_mapping = {
            "time": "時間",
            "gross_profit": "總利潤", # 這裡的映射很重要
            "trades": "交易筆數",
            "sharpe_ann": "Sharpe",
            "calmar_ann": "Calmar",
            "win_rate": "勝率",
            "profit_factor": "獲利因子",
            "positive_month_proportion": "正月比例",
            "avg_loss_negative_months": "負月平均虧損",
            "max_drawdown": "最大回撤",
            "iteration": "迭代次數",
            "best_score": "最佳得分",
            "seed": "隨機種子",
            "best_params": "最佳參數",
            "py_filename": "Python檔名"
        }
        self.csv_df.rename(columns=columns_mapping, inplace=True)

        # 更新下拉選單
        self.sort_column["values"] = list(self.csv_df.columns)
        self.secondary_sort_column["values"] = list(self.csv_df.columns)
        
        # 確保下拉選單有值可以選
        if len(self.csv_df.columns) > 0:
            # 嘗試設定預設選擇為 '總利潤'
            if "總利潤" in self.sort_column["values"]:
                self.sort_column.set("總利潤")
            else:
                self.sort_column.current(0) # 如果沒有總利潤，就選第一個

            self.secondary_sort_column.current(0)

        # 每次載入新檔案後，都嘗試以 '總利潤' 排序
        sort_by_col = "總利潤"
        if self.csv_df is not None and sort_by_col in self.csv_df.columns:
            self.csv_df = self.csv_df.sort_values(by=sort_by_col, ascending=False) # 預設遞減排序
            self.current_sort_col = sort_by_col
            self.is_ascending = False # 更新排序狀態
        
        self.update_tree(self.csv_df) # 更新 Treeview 顯示排序後的資料
        
        # 清空之前的標記，因為是新檔案
        self.tags = set()


    def sort_data(self):
        if self.csv_df is None:
            return
        primary_col = self.sort_column.get()
        secondary_col = self.secondary_sort_column.get()
        ascending = (self.order_var.get() == "遞增")

        if primary_col and secondary_col and primary_col != secondary_col:
            df_sorted = self.csv_df.sort_values(by=[primary_col, secondary_col],
                                                 ascending=[ascending, ascending])
        else:
            df_sorted = self.csv_df.sort_values(by=primary_col, ascending=ascending)

        # 更新畫面
        self.update_tree(df_sorted, preserve_tags=True)
        # 同時更新 self.csv_df，以便「更新當前 CSV」時會寫出最新版本
        self.csv_df = df_sorted

    def tag_selected(self):
        """手動標記：用 DataFrame 的 index 來記錄標記狀態"""
        selected_items = self.tree.selection()
        for item_id in selected_items:
            # item_id 就是我們在 update_tree 時用 str(idx) 做的
            df_idx = int(item_id)
            if df_idx in self.tags:
                self.tags.remove(df_idx)
                self.tree.item(item_id, tags=())
            else:
                self.tags.add(df_idx)
                self.tree.item(item_id, tags=("tagged",))

    def tag_selected_shortcut(self, event):
        self.tag_selected()

    def toggle_selection(self, event):
        """按空白鍵切換標記"""
        selected_items = self.tree.selection()
        for item_id in selected_items:
            df_idx = int(item_id)
            if df_idx in self.tags:
                self.tags.remove(df_idx)
                self.tree.item(item_id, tags=())
            else:
                self.tags.add(df_idx)
                self.tree.item(item_id, tags=("tagged",))

    def on_click_heading(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region == "heading":
            column_id = self.tree.identify_column(event.x)
            col_index = int(column_id.replace("#", "")) - 1
            col_name = self.tree["columns"][col_index]

            # 如果連續點同一欄，翻轉排序順序；不然就從頭開始
            if self.current_sort_col == col_name:
                self.is_ascending = not self.is_ascending
            else:
                self.current_sort_col = col_name
                self.is_ascending = True

            if self.csv_df is not None:
                df_sorted = self.csv_df.sort_values(by=col_name, ascending=self.is_ascending)
                self.update_tree(df_sorted, preserve_tags=True)
                self.csv_df = df_sorted

    def update_tree(self, df: pd.DataFrame, preserve_tags=False):
        """
        用 DataFrame 的 index 當作 TreeView item_id，顯示時數值自動到小數第2位
        並根據你的要求調整欄位寬度。
        """
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)

        # 定義固定欄位寬度
        fixed_column_widths = {
            "最佳參數": 12 * 12,  # 實際內容可能很長，強制固定寬度可能會導致內容不完全顯示
            "Python檔名": 4 * 12, 
            "時間": 16 * 12,    
            "隨機種子": 4 * 12,    
        }
        default_width = 9 * 12 

        # 設定欄位寬度與標頭
        for col in df.columns:
            width = fixed_column_widths.get(col, default_width)
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=width, stretch=False) # stretch=False 避免自動拉伸

        # 主要修改：顯示時四捨五入到小數第2位（只處理float/int欄位，其餘保持原狀）
        for idx, row in df.iterrows():
            formatted_row = []
            for value in row:
                if isinstance(value, (float, np.floating)):
                    formatted_row.append(f"{value:.2f}")
                elif isinstance(value, (int, np.integer)):
                    formatted_row.append(str(value))
                else:
                    formatted_row.append(str(value))
            item_id = str(idx)
            tags = ("tagged",) if preserve_tags and idx in self.tags else ()
            self.tree.insert("", "end", iid=item_id, values=formatted_row, tags=tags)

        # 標記樣式
        self.tree.tag_configure("tagged", background="#005577", foreground="#ffffff")


    def copy_to_clipboard(self, event):
        """雙擊複製最佳參數欄位的內容，將 .0 結尾的浮點數轉成整數"""
        selected_item = self.tree.selection()
        if selected_item:
            item_id = selected_item[0]
            col_names = self.tree["columns"]
            values = self.tree.item(item_id, "values")

            if values and "最佳參數" in col_names:
                col_index = col_names.index("最佳參數")
                param_value = values[col_index]

                # 嘗試解析成 list 並處理內容
                try:
                    parsed = ast.literal_eval(param_value)
                    if isinstance(parsed, list):
                        cleaned = [int(x) if isinstance(x, float) and x.is_integer() else x for x in parsed]
                        param_value = str(cleaned)
                except Exception:
                    pass  # 無法解析就保留原本的值

                self.master.clipboard_clear()
                self.master.clipboard_append(str(param_value))
                self.master.update()



def main():
    root = tk.Tk()
    root.state("zoomed")  # 預設全螢幕
    app = CSVSorterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
