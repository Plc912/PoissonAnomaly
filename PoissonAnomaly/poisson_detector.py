"""
泊松分布时间序列异常检测模块
适用于计数型时序数据（方差≈均值），如故障次数、事件发生次数等
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Dict, Optional, Any
import json
import pickle


class PoissonDetector:
    """
    基于泊松分布的时间序列异常检测器
    
    泊松分布适用于计数数据的异常检测，特别是方差≈均值的数据。
    泊松分布只有一个参数λ（均值=方差），适合描述单位时间内随机事件发生的次数。
    """
    
    def __init__(self, window_size: int = 50, threshold_percentile: float = 0.01):
        """
        初始化泊松分布异常检测器
        
        参数:
            window_size: int, 滑动窗口大小，用于拟合泊松分布
            threshold_percentile: float, 异常阈值（百分位数），
                               低于此概率的点被标记为异常，默认0.01（1%）
        """
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.lambda_ = None  # 泊松分布的λ参数（均值=方差）
        self.mean = None  # 分布均值
        self.var = None  # 分布方差
        self.threshold_low = None  # 下界阈值
        self.threshold_high = None  # 上界阈值
        self.is_fitted = False
        
    def fit(self, data: np.ndarray) -> 'PoissonDetector':
        """
        训练模型：拟合泊松分布参数
        
        参数:
            data: np.ndarray, 训练数据（一维非负整数数组）
            
        返回:
            self: 训练后的检测器对象
        """
        # 验证数据
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim != 1:
            raise ValueError("训练数据必须是一维数组")
            
        if np.any(data < 0):
            raise ValueError("泊松分布要求数据非负")
            
        if not np.all(np.equal(np.mod(data, 1), 0)):
            raise ValueError("泊松分布要求数据为整数（计数数据）")
            
        if len(data) < self.window_size:
            raise ValueError(f"训练数据长度 ({len(data)}) 必须大于等于窗口大小 ({self.window_size})")
        
        # 计算均值和方差
        self.mean = np.mean(data)
        self.var = np.var(data, ddof=1)
        
        # 检查是否适合泊松分布（方差≈均值）
        ratio = self.var / self.mean if self.mean > 0 else 0
        if ratio > 2.0:
            raise ValueError(
                f"数据方差/均值比率 ({ratio:.2f}) 远大于1，表明数据呈现过离散特征。"
                f"建议使用负二项分布而非泊松分布。"
            )
        
        # 泊松分布的参数λ等于均值
        self.lambda_ = self.mean
        
        # 计算异常阈值（基于累积分布函数）
        self.threshold_low = stats.poisson.ppf(self.threshold_percentile, self.lambda_)
        self.threshold_high = stats.poisson.ppf(1 - self.threshold_percentile, self.lambda_)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        预测异常：返回异常标签（1=异常，0=正常）
        
        参数:
            data: np.ndarray, 待检测数据（一维非负整数数组）
            
        返回:
            np.ndarray: 异常标签数组（1=异常，0=正常）
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit() 方法")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        if data.ndim != 1:
            raise ValueError("待检测数据必须是一维数组")
        
        # 计算每个点的累积概率
        cdf_values = stats.poisson.cdf(data, self.lambda_)
        
        # 标记异常：累积概率低于阈值百分位数或高于(1-阈值百分位数)
        anomalies = (cdf_values < self.threshold_percentile) | (cdf_values > (1 - self.threshold_percentile))
        
        return anomalies.astype(int)
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        计算异常分数：返回每个点的异常程度（概率质量函数值的负对数）
        
        参数:
            data: np.ndarray, 待检测数据（一维非负整数数组）
            
        返回:
            np.ndarray: 异常分数数组（负对数似然，值越大越异常）
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit() 方法")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        if data.ndim != 1:
            raise ValueError("待检测数据必须是一维数组")
        
        # 计算概率质量函数（PMF）
        pmf_values = stats.poisson.pmf(data, self.lambda_)
        
        # 使用负对数似然作为异常分数（避免log(0)）
        scores = -np.log(pmf_values + 1e-10)
        
        return scores
    
    def detect_with_details(self, data: np.ndarray) -> Dict[str, Any]:
        """
        检测异常并返回详细信息
        
        参数:
            data: np.ndarray, 待检测数据（一维非负整数数组）
            
        返回:
            dict: 包含异常标签、分数、统计信息等的字典
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit() 方法")
        
        predictions = self.predict(data)
        scores = self.score(data)
        cdf_values = stats.poisson.cdf(data, self.lambda_)
        
        anomaly_indices = np.where(predictions == 1)[0].tolist()
        
        result = {
            "total_points": len(data),
            "anomaly_count": int(np.sum(predictions)),
            "anomaly_rate": float(np.mean(predictions)),
            "anomaly_indices": anomaly_indices,
            "predictions": predictions.tolist(),
            "scores": scores.tolist(),
            "cdf_values": cdf_values.tolist(),
            "model_params": {
                "lambda": float(self.lambda_),
                "mean": float(self.mean),
                "variance": float(self.var),
                "variance_mean_ratio": float(self.var / self.mean) if self.mean > 0 else 0,
                "threshold_low": float(self.threshold_low),
                "threshold_high": float(self.threshold_high),
                "threshold_percentile": float(self.threshold_percentile)
            }
        }
        
        return result
    
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath: str, 保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无法保存")
        
        model_data = {
            "window_size": self.window_size,
            "threshold_percentile": self.threshold_percentile,
            "lambda_": self.lambda_,
            "mean": self.mean,
            "var": self.var,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "is_fitted": self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PoissonDetector':
        """
        从文件加载模型
        
        参数:
            filepath: str, 模型文件路径
            
        返回:
            PoissonDetector: 加载的检测器对象
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(
            window_size=model_data["window_size"],
            threshold_percentile=model_data["threshold_percentile"]
        )
        detector.lambda_ = model_data["lambda_"]
        detector.mean = model_data["mean"]
        detector.var = model_data["var"]
        detector.threshold_low = model_data["threshold_low"]
        detector.threshold_high = model_data["threshold_high"]
        detector.is_fitted = model_data["is_fitted"]
        
        return detector


class DataLoader:
    """数据加载器：支持CSV和TXT格式，具备智能列识别和自动转换功能"""
    
    # 时间列关键词
    TIME_KEYWORDS = ['time', 'timestamp', 'date', 'datetime', '时间', '日期']
    
    # 数值列关键词
    VALUE_KEYWORDS = ['value', 'count', 'num', 'amount', 'quantity', '数值', '计数', '数量']
    
    # 日志列关键词
    LOG_KEYWORDS = ['eventid', 'event', 'log', 'level', 'component', 'content', 'message']
    
    @staticmethod
    def _print_conversion_notice(original_cols: List[str], value_col: str, 
                                  time_col: Optional[str] = None, 
                                  conversion_type: str = "column_mapping"):
        """打印数据转换通知"""
        print("=" * 60)
        print("数据格式转换通知")
        print("=" * 60)
        
        if conversion_type == "column_mapping":
            print("数据格式已自动转换:")
            print(f"  '{value_col}' → 'value' 列")
            if time_col:
                print(f"  '{time_col}' → 'time' 列")
            print(f"原始列: {original_cols}")
        elif conversion_type == "log_aggregation":
            print("日志格式已自动聚合为计数时序:")
            print(f"  事件列: '{value_col}'")
            if time_col:
                print(f"  时间列: '{time_col}'")
            print(f"  聚合方式: 按时间窗口统计事件发生次数")
            print(f"原始列: {original_cols}")
        
        print("=" * 60)
    
    @staticmethod
    def _find_column(df: pd.DataFrame, keywords: List[str], 
                     col_type: str = "column") -> Optional[str]:
        """
        根据关键词查找列名
        
        参数:
            df: DataFrame
            keywords: 关键词列表
            col_type: 列类型（用于错误提示）
        
        返回:
            str or None: 找到的列名，未找到返回None
        """
        # 先精确匹配
        for col in df.columns:
            if col.lower() in keywords:
                return col
        
        # 再模糊匹配（包含关键词）
        for col in df.columns:
            col_lower = col.lower()
            for keyword in keywords:
                if keyword in col_lower:
                    return col
        
        return None
    
    @staticmethod
    def _find_numeric_column(df: pd.DataFrame) -> Optional[str]:
        """查找数值类型的列"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # 优先选择整数类型
            for col in numeric_cols:
                if df[col].dtype in [np.int32, np.int64]:
                    return col
            # 其次选择第一个数值列
            return numeric_cols[0]
        return None
    
    @staticmethod
    def _is_log_format(df: pd.DataFrame) -> bool:
        """判断是否为日志格式"""
        log_col_count = 0
        for col in df.columns:
            if col.lower() in DataLoader.LOG_KEYWORDS:
                log_col_count += 1
        
        # 如果有2个以上日志相关列，认为是日志格式
        return log_col_count >= 2
    
    @staticmethod
    def _aggregate_log_to_counts(df: pd.DataFrame, time_col: str, 
                                  event_col: str, time_window: str = "1min") -> Tuple[np.ndarray, pd.Series]:
        """
        将日志格式数据聚合为计数时序
        
        参数:
            df: 原始日志DataFrame
            time_col: 时间列名
            event_col: 事件列名
            time_window: 时间窗口（如 '1min', '5min', '1H'）
        
        返回:
            tuple: (计数数组, 时间戳序列)
        """
        # 解析时间列
        df_copy = df.copy()
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        
        # 删除无效时间戳
        df_copy = df_copy.dropna(subset=[time_col])
        
        # 按时间窗口分组统计
        df_copy = df_copy.set_index(time_col)
        counts = df_copy.resample(time_window).size()
        
        return counts.values.astype(int), counts.index
    
    @staticmethod
    def load_csv(filepath: str, 
                 value_column: Optional[str] = None, 
                 timestamp_column: Optional[str] = None, 
                 nrows: Optional[int] = None,
                 autoconvert: bool = True,
                 time_window: str = "1min") -> Tuple[np.ndarray, Optional[pd.Series]]:
        """
        从CSV文件加载时间序列数据（支持智能格式识别和转换）
        
        参数:
            filepath: str, CSV文件路径
            value_column: Optional[str], 数值列名（None时自动识别）
            timestamp_column: Optional[str], 时间戳列名（None时自动识别）
            nrows: Optional[int], 加载的行数限制
            autoconvert: bool, 是否自动转换数据格式，默认True
            time_window: str, 日志数据的时间聚合窗口（如 '1min', '5min', '1H'）
            
        返回:
            tuple: (数据数组, 时间戳序列或None)
            
        支持格式:
            1. 标准格式：包含 'time' 和 'value' 列
            2. 自定义列名：自动识别时间列和数值列
            3. 日志格式：自动聚合为计数时序
        """
        df = pd.read_csv(filepath, nrows=nrows)
        original_columns = df.columns.tolist()
        
        # 情况1：用户明确指定了列名
        if value_column and value_column in df.columns:
            data = df[value_column].values
            # 转换为整数
            data = np.round(data).astype(int)
            
            timestamps = None
            if timestamp_column and timestamp_column in df.columns:
                timestamps = pd.to_datetime(df[timestamp_column])
            
            return data, timestamps
        
        # 情况2：autoconvert=False 且未找到指定列
        if not autoconvert:
            if value_column:
                raise ValueError(f"CSV文件中未找到列 '{value_column}'")
            else:
                raise ValueError("请指定 'value_column' 或启用 'autoconvert=True'")
        
        # 情况3：autoconvert=True，开始智能识别
        
        # 3.1 检查是否为日志格式
        if DataLoader._is_log_format(df):
            time_col = DataLoader._find_column(df, DataLoader.TIME_KEYWORDS, "time")
            event_col = DataLoader._find_column(df, DataLoader.LOG_KEYWORDS, "event")
            
            if time_col and event_col:
                data, timestamps = DataLoader._aggregate_log_to_counts(
                    df, time_col, event_col, time_window
                )
                DataLoader._print_conversion_notice(
                    original_columns, event_col, time_col, "log_aggregation"
                )
                return data, timestamps
        
        # 3.2 自动识别数值列
        detected_value_col = None
        
        # 先查找标准列名
        if 'value' in df.columns:
            detected_value_col = 'value'
        else:
            # 根据关键词查找
            detected_value_col = DataLoader._find_column(df, DataLoader.VALUE_KEYWORDS, "value")
            
            # 如果还没找到，查找数值类型的列
            if not detected_value_col:
                detected_value_col = DataLoader._find_numeric_column(df)
        
        if not detected_value_col:
            raise ValueError(
                f"无法自动识别数值列。请明确指定 'value_column' 参数。\n"
                f"可用列: {original_columns}"
            )
        
        # 3.3 自动识别时间列
        detected_time_col = None
        if 'time' in df.columns:
            detected_time_col = 'time'
        else:
            detected_time_col = DataLoader._find_column(df, DataLoader.TIME_KEYWORDS, "time")
        
        # 3.4 提取数据
        data = df[detected_value_col].values
        data = np.round(data).astype(int)
        
        timestamps = None
        if detected_time_col:
            timestamps = pd.to_datetime(df[detected_time_col], errors='coerce')
        
        # 3.5 显示转换通知
        if detected_value_col != 'value' or detected_time_col:
            DataLoader._print_conversion_notice(
                original_columns, detected_value_col, detected_time_col, "column_mapping"
            )
        
        return data, timestamps
    
    @staticmethod
    def load_txt(filepath: str, delimiter: str = ' ', 
                 nrows: Optional[int] = None) -> np.ndarray:
        """
        从TXT文件加载时间序列数据
        
        参数:
            filepath: str, TXT文件路径
            delimiter: str, 分隔符，默认空格
            nrows: Optional[int], 加载的行数限制
            
        返回:
            np.ndarray: 数据数组（一维整数）
        """
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if nrows and i >= nrows:
                    break
                line = line.strip()
                if line:
                    values = [int(float(x)) for x in line.split(delimiter) if x]
                    data.extend(values)
        
        return np.array(data, dtype=int)
    
    @staticmethod
    def save_csv(filepath: str, data: np.ndarray, 
                 predictions: Optional[np.ndarray] = None,
                 scores: Optional[np.ndarray] = None,
                 timestamps: Optional[pd.Series] = None) -> None:
        """
        保存检测结果到CSV文件
        
        参数:
            filepath: str, 保存路径
            data: np.ndarray, 原始数据
            predictions: Optional[np.ndarray], 异常预测（0/1）
            scores: Optional[np.ndarray], 异常分数
            timestamps: Optional[pd.Series], 时间戳
        """
        result_df = pd.DataFrame({"value": data})
        
        if timestamps is not None:
            result_df.insert(0, "timestamp", timestamps)
        
        if predictions is not None:
            result_df["is_anomaly"] = predictions
        
        if scores is not None:
            result_df["anomaly_score"] = scores
        
        result_df.to_csv(filepath, index=False)

