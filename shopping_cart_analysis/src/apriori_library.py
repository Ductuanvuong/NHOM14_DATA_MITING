# -*- coding: utf-8 -*-
"""
Shopping Cart Library

This library contains classes for data cleaning, feature engineering,
and association rule analysis for shopping cart.
"""

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import networkx as nx


# =========================================================
# 1. DATA CLEANER
# =========================================================

class DataCleaner:
    """
    A class for cleaning and preprocessing retail transaction data.

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path):
        """
        Initialize the DataCleaner with data path.

        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        """
        Load and display basic information about the dataset.

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Chuyển CustomerID thành format 6 ký tự
        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

        print(f"Kích thước dữ liệu: {self.df.shape}")
        print(f"Số bản ghi: {len(self.df):,}")

        return self.df

    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing on UK customers.

        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Thêm cột TotalPrice
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Loại bỏ các hóa đơn bị hủy (bắt đầu bằng 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # Chỉ tập trung vào khách hàng UK
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Loại bỏ các sản phẩm có quantity hoặc price không hợp lệ
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        # Bỏ description NA
        self.df_uk = self.df_uk.dropna(subset=["Description"])

        return self.df_uk

    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def add_total_price(self):
        """
        Add TotalPrice column (Quantity * UnitPrice) to cleaned UK data.
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        self.df_uk["TotalPrice"] = self.df_uk["Quantity"] * self.df_uk["UnitPrice"]
        return self.df_uk

    def compute_rfm(self, snapshot_date=None):
        """
        Compute RFM (Recency, Frequency, Monetary) for each customer based on cleaned UK data.

        Args:
            snapshot_date (datetime or str, optional):
                Reference date for Recency calculation.
                - If None: use max(InvoiceDate) + 1 day.

        Returns:
            pd.DataFrame: RFM dataframe with columns [CustomerID, Recency, Frequency, Monetary]
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        df = self.df_uk.copy()

        # Đảm bảo có TotalPrice
        if "TotalPrice" not in df.columns:
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

        # Xác định snapshot_date
        if snapshot_date is None:
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
        else:
            # Cho phép truyền vào dạng string 'YYYY-MM-DD'
            if isinstance(snapshot_date, str):
                snapshot_date = pd.to_datetime(snapshot_date)

        # Tính RFM
        rfm = df.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": "nunique",  # Frequency
                "TotalPrice": "sum",     # Monetary
            }
        )

        rfm.rename(
            columns={
                "InvoiceDate": "Recency",
                "InvoiceNo": "Frequency",
                "TotalPrice": "Monetary",
            },
            inplace=True,
        )

        self.rfm_data = rfm.reset_index()
        return self.rfm_data

    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.

        Args:
            output_dir (str): Output directory path
        """
        if self.df_uk is None:
            raise ValueError("Cleaned UK data not available. Call clean_data() first.")

        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/cleaned_uk_data.csv"
        self.df_uk.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đã làm sạch: {output_path}")


# =========================================================
# 2. BASKET PREPARER
# =========================================================

class BasketPreparer:
    """
    A class for preparing basket data for association rule mining.

    This class transforms transaction data into a format suitable for
    applying the Apriori algorithm.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        invoice_col: str = "InvoiceNo",
        item_col: str = "Description",
        quantity_col: str = "Quantity",
    ):
        """
        Initialize the BasketPreparer with cleaned dataframe.

        Args:
            df (pd.DataFrame): Cleaned transaction-level dataframe
            invoice_col (str): Column name for invoice number
            item_col (str): Column name for item description
            quantity_col (str): Column name for item quantity
        """
        self.df = df
        self.invoice_col = invoice_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.basket = None
        self.basket_bool = None

    def create_basket(self):
        """
        Create a basket format dataframe for Apriori algorithm.

        Returns:
            pd.DataFrame: Basket format dataframe
        """

        basket = (
            self.df.groupby([self.invoice_col, self.item_col])[self.quantity_col]
            .sum()
            .unstack()
            .fillna(0)
        )

        self.basket = basket
        return self.basket

    def encode_basket(self, threshold: int = 1):
        """
        Encode the basket dataframe into boolean format.

        Args:
            threshold (int): Minimum quantity to consider an item as present

        Returns:
            pd.DataFrame: Boolean encoded basket dataframe
        """

        if self.basket is None:
            raise ValueError("Basket not created. Please run create_basket() first.")
        basket_bool = self.basket.applymap(lambda x: 1 if x >= threshold else 0)
        basket_bool = basket_bool.astype(bool)
        self.basket_bool = basket_bool
        return self.basket_bool

    def save_basket_bool(self, output_path: str):
        """
        Save the boolean encoded basket dataframe to a Parquet file.

        Args:
            output_path (str): Path to save the Parquet file
        """
        if self.basket_bool is None:
            raise ValueError("Basket not encoded. Please call encode_basket() first.")
        basket_bool_to_save = self.basket_bool.reset_index(drop=True)

        basket_bool_to_save.to_parquet(output_path, index=False)
        print(f"Đã lưu basket boolean: {output_path}")


# =========================================================
# 3. APRIORI ASSOCIATION RULES MINER
# =========================================================

class AssociationRulesMiner:
    """
    A class for mining association rules using the Apriori algorithm.

    This class applies the Apriori algorithm to the basket data and extracts
    association rules based on specified metrics.
    """

    def __init__(self, basket_bool: pd.DataFrame):
        """
        Initialize the AssociationRulesMiner with basket data.

        Args:
            basket_bool (pd.DataFrame): Boolean encoded basket dataframe
        """
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None

    def mine_frequent_itemsets(
        self,
        min_support: float = 0.01,
        max_len: int = None,
        use_colnames: bool = True,
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using the Apriori algorithm.

        Returns:
            pd.DataFrame: DataFrame of frequent itemsets
        """

        fi = apriori(
            self.basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )

        fi.sort_values(by="support", ascending=False, inplace=True)
        self.frequent_itemsets = fi
        return self.frequent_itemsets

    def generate_rules(
        self,
        metric: str = "lift",
        min_threshold: float = 1.0,
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets.

        Args:
            metric (str): Metric to evaluate the rules
            min_threshold (float): Minimum threshold for the metric

        Returns:
            pd.DataFrame: DataFrame of association rules
        """

        if self.frequent_itemsets is None:
            raise ValueError(
                "Frequent itemsets not mined. Please run mine_frequent_itemsets() first."
            )

        rules = association_rules(
            self.frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold,
        )

        rules = rules.sort_values(["lift", "confidence"], ascending=False)
        self.rules = rules
        return self.rules

    @staticmethod
    def _frozenset_to_str(fs: frozenset) -> str:
        return ", ".join(sorted(list(fs)))

    def add_readable_rule_str(self) -> pd.DataFrame:
        """
        Add human-readable columns for antecedents, consequents, and rule_str
        to the rules dataframe.

        Returns:
            pd.DataFrame: Rules dataframe with extra readable columns
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        rules = self.rules.copy()
        rules["antecedents_str"] = rules["antecedents"].apply(self._frozenset_to_str)
        rules["consequents_str"] = rules["consequents"].apply(self._frozenset_to_str)
        rules["rule_str"] = rules["antecedents_str"] + " → " + rules["consequents_str"]

        self.rules = rules
        return self.rules

    def filter_rules(
        self,
        min_support: float = None,
        min_confidence: float = None,
        min_lift: float = None,
        max_len_antecedents: int = None,
        max_len_consequents: int = None,
    ) -> pd.DataFrame:
        """
        Filter rules based on support, confidence, lift and length of antecedents/consequents.
        """
        if self.rules is None:
            raise ValueError("rules is not available. Call generate_rules() first.")

        filtered = self.rules.copy()

        if min_support is not None:
            filtered = filtered[filtered["support"] >= min_support]
        if min_confidence is not None:
            filtered = filtered[filtered["confidence"] >= min_confidence]
        if min_lift is not None:
            filtered = filtered[filtered["lift"] >= min_lift]
        if max_len_antecedents is not None:
            filtered = filtered[
                filtered["antecedents"].apply(len) <= max_len_antecedents
            ]
        if max_len_consequents is not None:
            filtered = filtered[
                filtered["consequents"].apply(len) <= max_len_consequents
            ]

        filtered = filtered.reset_index(drop=True)
        return filtered

    def save_rules(self, output_path: str, rules_df: pd.DataFrame = None):
        """
        Save rules dataframe to CSV.

        Args:
            output_path (str): CSV path
            rules_df (pd.DataFrame): Rules dataframe to save (if None, use self.rules)
        """
        if rules_df is None:
            if self.rules is None:
                raise ValueError("No rules to save.")
            rules_df = self.rules

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rules_df.to_csv(output_path, index=False)
        print(f"Đã lưu luật vào: {output_path}")


# =========================================================
# 4. DATA VISUALIZER (EDA + RFM + APRIORI)
# =========================================================

class DataVisualizer:
    """
    A class for creating visualizations for customer segmentation and
    shopping behavior analysis.

    This class provides methods for plotting various aspects of the data
    including temporal patterns, customer behavior, RFM analysis,
    và trực quan hoá luật kết hợp (Apriori).
    """

    def __init__(self):
        """Initialize the DataVisualizer with plotting settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Doanh thu hàng ngày")
        plt.xlabel("Ngày")
        plt.ylabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Doanh thu hàng tháng")
        plt.xlabel("Tháng")
        plt.ylabel("Doanh thu (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.

        Args:
            df (pd.DataFrame): Dataframe with time features:
                DayOfWeek, HourOfDay
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Hoạt động mua hàng theo ngày và giờ")
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe (có Quantity, TotalPrice)
            top_n (int): Number of top products to show
        """
        # Top sản phẩm theo số lượng
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} sản phẩm theo số lượng bán")
        plt.xlabel("Số lượng bán")
        plt.tight_layout()
        plt.show()

        # Top sản phẩm theo doanh thu
        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} sản phẩm theo doanh thu")
        plt.xlabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.

        Args:
            df (pd.DataFrame): Transaction dataframe with CustomerID, InvoiceNo, TotalPrice
        """
        # Số giao dịch trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Phân phối số giao dịch trên mỗi khách hàng")
        plt.xlabel("Số giao dịch")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

        # Chi tiêu trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Phân phối tổng chi tiêu trên mỗi khách hàng")
        plt.xlabel("Tổng chi tiêu (GBP)")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data (pd.DataFrame): RFM dataframe with
                columns ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        """
        # RFM distributions
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Phân phối Recency (Ngày kể từ lần mua cuối)")
        axes[0].set_xlabel("Ngày")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Phân phối Frequency (Số giao dịch)")
        axes[1].set_xlabel("Số giao dịch")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Phân phối Monetary (Tổng chi tiêu)")
        axes[2].set_xlabel("Tổng chi tiêu (GBP)")

        plt.tight_layout()
        plt.show()

# Apriori visualizations

    @staticmethod
    def _itemset_to_str(itemset):
        """
        Chuyển một itemset (frozenset, set, list, tuple) thành chuỗi có thể đọc được.

        Args:
            itemset: tập mục dưới dạng tập, danh sách, frozenset, v.v.
        """
        if isinstance(itemset, (set, frozenset, list, tuple)):
            return ", ".join(sorted(map(str, itemset)))
        return str(itemset)

    def plot_top_frequent_itemsets(
        self,
        frequent_itemsets: pd.DataFrame,
        top_n: int = 20,
        min_len: int | None = None,
        max_len: int | None = None,
        title: str = "Top frequent itemsets theo support",
    ):
        """
        Vẽ biểu đồ cột thể hiện các tập mục phổ biến nhất theo support.

        Args:
            frequent_itemsets: DataFrame kết quả từ mlxtend.frequent_patterns.apriori
                với tối thiểu hai cột 'itemsets' và 'support'.
            top_n: số lượng itemset hiển thị.
            min_len: chỉ lấy các itemset có độ dài >= min_len (nếu không None).
            max_len: chỉ lấy các itemset có độ dài <= max_len (nếu không None).
            title: tiêu đề biểu đồ.
        """
        if "itemsets" not in frequent_itemsets.columns or "support" not in frequent_itemsets.columns:
            raise ValueError("frequent_itemsets cần có cột 'itemsets' và 'support'.")

        fi = frequent_itemsets.copy()

        if min_len is not None:
            fi = fi[fi["itemsets"].apply(len) >= min_len]
        if max_len is not None:
            fi = fi[fi["itemsets"].apply(len) <= max_len]

        fi = fi.sort_values("support", ascending=False).head(top_n).copy()
        if fi.empty:
            print("Không có itemset nào thỏa mãn điều kiện lọc.")
            return

        fi["itemset_str"] = fi["itemsets"].apply(self._itemset_to_str)

        plt.figure(figsize=(12, max(4, 0.4 * len(fi))))
        sns.barplot(data=fi, x="support", y="itemset_str")
        plt.title(title)
        plt.xlabel("Support")
        plt.ylabel("Itemset")
        plt.tight_layout()
        plt.show()

    def plot_itemset_length_distribution(
        self,
        frequent_itemsets: pd.DataFrame,
        title: str = "Phân phối độ dài các tập mục (itemset length)",
    ):
        """
        Vẽ phân phối số lượng itemset theo độ dài (1-itemset, 2-itemset, ...).

        Args:
            frequent_itemsets: DataFrame kết quả từ apriori() với cột 'itemsets'.
            title: tiêu đề biểu đồ.
        """
        if "itemsets" not in frequent_itemsets.columns:
            raise ValueError("frequent_itemsets cần có cột 'itemsets'.")

        lengths = frequent_itemsets["itemsets"].apply(len)
        length_counts = lengths.value_counts().sort_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(x=length_counts.index, y=length_counts.values)
        plt.title(title)
        plt.xlabel("Độ dài itemset")
        plt.ylabel("Số lượng itemset")
        plt.xticks(length_counts.index)
        plt.tight_layout()
        plt.show()

    def plot_top_rules_bar(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        sort_by: str = "lift",
        title: str = "Top luật kết hợp",
    ):
        """
        Vẽ biểu đồ cột thể hiện top_n luật kết hợp theo một metric (lift/confidence/support).

        Args:
            rules_df: DataFrame kết quả từ association_rules() và đã có cột 'rule_str'.
            top_n: số luật hiển thị.
            sort_by: cột dùng để sắp xếp ('lift', 'confidence', 'support', ...).
            title: tiêu đề chung của biểu đồ.
        """
        if "rule_str" not in rules_df.columns:
            raise ValueError("rules_df cần có cột 'rule_str' (gọi add_readable_rule_str() trước).")
        if sort_by not in rules_df.columns:
            raise ValueError(f"rules_df không có cột '{sort_by}' để sắp xếp.")

        df = rules_df.sort_values(sort_by, ascending=False).head(top_n).copy()
        if df.empty:
            print("Không có luật nào để vẽ.")
            return

        plt.figure(figsize=(12, max(4, 0.4 * len(df))))
        sns.barplot(data=df, x=sort_by, y="rule_str")
        plt.title(f"{title} (theo {sort_by}) - Top {len(df)} luật")
        plt.xlabel(sort_by.capitalize())
        plt.ylabel("Luật (antecedent → consequent)")
        plt.tight_layout()
        plt.show()

    def plot_top_rules_lift(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        title_prefix: str = "Top luật theo Lift (Apriori)",
    ):
        """
        Vẽ biểu đồ top luật theo chỉ số Lift.

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            top_n: số luật lấy top theo lift.
            title_prefix: phần tiêu đề, sẽ gắn thêm số luật thực tế.
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ top lift.")
            return

        self.plot_top_rules_bar(
            rules_df=rules_df,
            top_n=top_n,
            sort_by="lift",
            title=title_prefix,
        )

    def plot_top_rules_confidence(
        self,
        rules_df: pd.DataFrame,
        top_n: int = 20,
        title_prefix: str = "Top luật theo Confidence (Apriori)",
    ):
        """
        Vẽ biểu đồ top luật theo chỉ số Confidence (tương ứng code gốc ở cell 17).

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            top_n: số luật lấy top theo confidence.
            title_prefix: phần tiêu đề, sẽ gắn thêm số luật thực tế.
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ top confidence.")
            return

        self.plot_top_rules_bar(
            rules_df=rules_df,
            top_n=top_n,
            sort_by="confidence",
            title=title_prefix,
        )

    def plot_rules_support_confidence_scatter(
        self,
        rules_df: pd.DataFrame,
        title: str = "Phân bố luật: Support vs Confidence (màu = Lift)",
        point_size: int = 40,
    ):
        """
        Vẽ scatter plot Support–Confidence, màu theo Lift

        Args:
            rules_df: DataFrame, thường là rules_filtered_ap.
            title: tiêu đề biểu đồ.
            point_size: kích thước điểm (tham số s của matplotlib).
        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ scatter.")
            return

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            rules_df["support"],
            rules_df["confidence"],
            c=rules_df["lift"],
            s=point_size,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Lift")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_pairwise_lift_heatmap(
        self,
        rules_df: pd.DataFrame,
        top_items: int = 15,
        metric: str = "lift",
        title: str = "Heatmap lift giữa các cặp sản phẩm (1→1)",
    ):
        """
        Vẽ heatmap lift (hoặc metric khác) cho các luật 1 sản phẩm → 1 sản phẩm.

        Args:
            rules_df: DataFrame kết quả từ association_rules() + add_readable_rule_str().
            top_items: số lượng sản phẩm phổ biến nhất xét đến (theo tần suất xuất hiện trong luật).
            metric: tên cột để vẽ (thường là 'lift' hoặc 'confidence').
            title: tiêu đề biểu đồ.
        """
        required_cols = {"antecedents", "consequents", metric}
        if not required_cols.issubset(set(rules_df.columns)):
            raise ValueError(f"rules_df cần có các cột: {required_cols}")

        # Chỉ giữ các luật 1 sản phẩm → 1 sản phẩm
        single_rules = rules_df[
            (rules_df["antecedents"].apply(len) == 1)
            & (rules_df["consequents"].apply(len) == 1)
        ].copy()

        if single_rules.empty:
            print("Không có luật 1→1 nào để vẽ heatmap.")
            return

        # Tạo tên sản phẩm dạng chuỗi
        single_rules["antecedent_str"] = single_rules["antecedents"].apply(
            lambda x: list(x)[0]
        )
        single_rules["consequent_str"] = single_rules["consequents"].apply(
            lambda x: list(x)[0]
        )

        # Lấy top_items sản phẩm xuất hiện nhiều nhất trong luật
        all_items = pd.concat(
            [single_rules["antecedent_str"], single_rules["consequent_str"]]
        )
        top_item_names = all_items.value_counts().head(top_items).index

        df_filtered = single_rules[
            single_rules["antecedent_str"].isin(top_item_names)
            & single_rules["consequent_str"].isin(top_item_names)
        ]

        if df_filtered.empty:
            print("Sau khi lọc top_items, không còn luật nào để vẽ heatmap.")
            return

        pivot = df_filtered.pivot_table(
            index="antecedent_str",
            columns="consequent_str",
            values=metric,
            aggfunc="max",
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
        )
        plt.title(title + f" (metric = {metric})")
        plt.xlabel("Consequent")
        plt.ylabel("Antecedent")
        plt.tight_layout()
        plt.show()
    def plot_rules_support_confidence_scatter_interactive(
        self,
        rules_df: pd.DataFrame,
        title: str = "Biểu đồ tương tác: Support vs Confidence (màu & kích thước = Lift)"
    ):
        """
        Biểu đồ scatter tương tác bằng Plotly:
        - Trục X: support
        - Trục Y: confidence
        - Màu & kích thước điểm: lift
        - hover hiển thị rule_str

        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ scatter Plotly.")
            return

        # Đảm bảo có rule_str (nếu chưa thì gợi ý)
        if "rule_str" not in rules_df.columns:
            print("rules_df chưa có cột 'rule_str'. Hãy gọi miner.add_readable_rule_str() trước.")
            return

        fig = px.scatter(
            rules_df,
            x="support",
            y="confidence",
            color="lift",
            size="lift",
            hover_name="rule_str",
            title=title,
            labels={
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
            },
        )
        fig.show()

    def plot_rules_network(
        self,
        rules_df: pd.DataFrame,
        max_rules: int | None = 100,
        min_lift: float | None = None,
        title: str = "Mạng lưới các luật kết hợp (Arrow: antecedent → consequent)",
        figsize: tuple = (12, 8),
    ):
        """
        Vẽ network graph các luật kết hợp bằng networkx:
        - Node: sản phẩm
        - Edge có hướng: antecedent -> consequent
        - Độ dày cạnh tỷ lệ với lift

        """
        if rules_df is None or rules_df.empty:
            print("Không có luật nào sau khi lọc để vẽ network graph.")
            return

        required_cols = {"antecedents", "consequents", "lift"}
        if not required_cols.issubset(rules_df.columns):
            raise ValueError(f"rules_df cần có các cột: {required_cols}")

        # Lọc theo lift nếu có
        df = rules_df.copy()
        if min_lift is not None:
            df = df[df["lift"] >= min_lift]

        if df.empty:
            print("Không còn luật nào sau khi lọc theo min_lift để vẽ network graph.")
            return

        # Giới hạn số luật để network không quá rối
        if max_rules is not None:
            df = df.sort_values("lift", ascending=False).head(max_rules)

        G = nx.DiGraph()

        # Tạo node + edge
        edges = []
        for _, row in df.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            lift_value = row["lift"]

            for a in antecedents:
                for c in consequents:
                    G.add_node(a)
                    G.add_node(c)
                    G.add_edge(a, c, weight=lift_value)
                    edges.append((a, c, lift_value))

        if not edges:
            print("Không tạo được cạnh nào cho network graph.")
            return

        # Layout
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Tính độ dày cạnh
        weights = [w for (_, _, w) in edges]
        max_w = max(weights)
        norm_widths = [w / max_w * 2 for w in weights]  # scale về khoảng [0, 2]

        # Vẽ node
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
        # Vẽ label
        nx.draw_networkx_labels(G, pos, font_size=9)

        # Vẽ edge có hướng
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=15,
            width=norm_widths,
            edge_color="gray",
        )

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
