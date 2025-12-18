# 泊松分布时间序列异常检测 MCP 工具

作者：庞力铖

邮箱：3522236586@qq.com

GitHub:

基于泊松分布的时间序列异常检测工具，专门针对**计数型时序数据**（方差≈均值）进行异常检测，如故障次数、事件发生次数等。本工具封装为MCP（Model Context Protocol）服务器，可与支持MCP的AI助手集成使用。

## 项目简介

泊松分布是一种离散概率分布，适合描述单位时间内随机事件发生的次数。泊松分布只有一个参数λ（均值=方差），当数据方差≈均值时非常适用。

**适用场景：**

- 系统故障次数异常检测
- 服务器错误日志计数监控
- 网站访问次数异常识别（低频场景）
- API调用失败次数异常检测
- 客服投诉次数异常监控
- 其他方差≈均值的计数时序数据异常检测

**为什么使用泊松分布？**

- **简单高效**：只有一个参数λ，拟合和计算速度快
- **方差=均值**：适合描述随机事件发生次数
- **理论基础扎实**：经典概率分布，广泛应用于质量控制、可靠性分析
- **适合低频事件**：故障、错误等低频事件通常符合泊松分布

## 核心特性

- **基于泊松分布**：专门针对方差≈均值的计数数据优化
- **异步任务管理**：支持后台训练和检测，实时查询进度
- **灵活的数据输入**：支持CSV和TXT格式
- **模型持久化**：可保存和加载训练好的模型
- **详细的检测结果**：返回异常标签、异常分数、概率值、方差均值比等
- **MCP协议封装**：可与Claude等AI助手无缝集成
- **自动验证**：检查数据是否适合泊松分布

## 安装依赖

```bash
pip install -r requirements.txt
```

### 1. 启动MCP服务器

```bash
python poisson_mcp_server.py
```

服务器将在端口 `2252` 上启动（使用SSE传输）。

### 2. 使用MCP工具

服务器提供以下MCP工具：

#### 2.1 训练模型 (`poisson_train`)

训练泊松分布异常检测模型：

**参数：**

- `csv`: CSV文件路径（与 `txt`二选一）
- `txt`: TXT文件路径（与 `csv`二选一）
- `value_column`: CSV中的数值列名，默认 `"value"`
- `window_size`: 滑动窗口大小，默认 `50`
- `threshold_percentile`: 异常阈值百分位数（0-1），默认 `0.01`（1%）
- `nrows`: 可选，仅加载前n行
- `save`: 可选，保存模型的路径

**返回：**

```json
{
  "status": "queued",
  "task_id": "uuid",
  "type": "train"
}
```

**示例：**

```python
# 从CSV训练
poisson_train(csv="failures.csv", value_column="failure_count", save="model.pkl")

# 从TXT训练
poisson_train(txt="data.txt", window_size=100, save="model.pkl")
```

#### 2.2 异常检测 (`poisson_detect`)

使用已训练的模型进行异常检测：

**参数：**

- `model_path`: 模型文件路径（必需）
- `csv`: CSV文件路径（与 `txt`二选一）
- `txt`: TXT文件路径（与 `csv`二选一）
- `value_column`: CSV中的数值列名，默认 `"value"`
- `nrows`: 可选，仅加载前n行
- `limit`: 返回数据点数量限制，默认 `1000`
- `save_result`: 可选，保存检测结果的CSV路径

**返回：**

```json
{
  "status": "queued",
  "task_id": "uuid",
  "type": "detect"
}
```

**检测结果包含：**

- `total_points`: 总数据点数
- `anomaly_count`: 异常点数量
- `anomaly_rate`: 异常率
- `anomaly_indices`: 异常点索引列表
- `predictions`: 异常标签数组（0=正常，1=异常）
- `scores`: 异常分数数组（负对数似然）
- `cdf_values`: 累积概率值数组
- `model_params`: 模型参数
  - `lambda`: 泊松分布参数λ（均值=方差）
  - `mean`: 分布均值
  - `variance`: 分布方差
  - `variance_mean_ratio`: 方差均值比（应≈1）
  - `threshold_low`: 下界阈值
  - `threshold_high`: 上界阈值

**示例：**

```python
# 检测异常并保存结果
poisson_detect(model_path="model.pkl", csv="test_data.csv", save_result="result.csv")
```

#### 2.3 任务管理

**列出所有任务 (`list_tasks`)**

```python
list_tasks()
```

**获取任务详情 (`get_task`)**

```python
get_task(task_id="uuid")
```

**取消任务 (`cancel_task`)**

```python
cancel_task(task_id="uuid")
```

**清除已完成任务 (`clear_tasks`)**

```python
clear_tasks()
```

## 数据格式

### 🎯 智能数据格式识别（新功能）

本工具现在支持**智能列识别**和**自动数据转换**功能！无需手动指定列名，AI会自动识别并转换数据格式。

#### 支持的格式类型

**1. 标准格式（无需转换）**

包含 `time` 和 `value` 列：

```csv
time,value
2024-01-01 00:00:00,2
2024-01-01 01:00:00,1
2024-01-01 02:00:00,8
```

**2. 自定义列名格式（自动转换）**

自动识别包含以下关键词的列：
- **时间列**：time, timestamp, date, datetime, 时间, 日期
- **数值列**：value, count, num, amount, quantity, 数值, 计数, 数量
- 或任何数值类型的列

示例：
```csv
order_date,order_count,shop_id
2024-01-01 00:00:00,15,shop_001
2024-01-01 01:00:00,12,shop_001
2024-01-01 02:00:00,8,shop_001
```

**3. 日志格式（自动聚合为计数时序）**

包含日志相关列（EventId, Event, Log, Level, Component, Content, Message等）时，自动按时间窗口聚合为计数数据：

```csv
timestamp,EventId,Level,Component,Message
2024-01-01 00:00:10,E01,INFO,API,Request received
2024-01-01 00:00:25,E02,ERROR,DB,Connection failed
2024-01-01 00:00:40,E01,INFO,API,Request received
```

聚合后：每个时间窗口的事件发生次数

#### 使用参数

**autoconvert: bool = True**
- 是否自动转换数据格式
- 当数据不符合标准格式时，AI会自动识别并转换
- 默认启用

**time_window: str = "1min"**
- 日志数据的时间聚合窗口
- 支持：'1min', '5min', '1H', '1D' 等
- 仅在日志格式时有效

#### 转换通知

当检测到数据需要转换时，系统会自动显示：

```
============================================================
数据格式转换通知
============================================================
数据格式已自动转换:
  'order_count' → 'value' 列
  'order_date' → 'time' 列
原始列: ['order_date', 'order_count', 'shop_id']
============================================================
```

#### 使用示例

```python
# 自动识别列名
poisson_train(
    csv="orders.csv",      # 包含 order_count, order_date 列
    autoconvert=True,      # 自动识别和转换
    save="model.pkl"
)

# 日志格式自动聚合
poisson_train(
    csv="system_logs.csv", # 日志格式数据
    time_window="5min",    # 按5分钟聚合
    autoconvert=True,
    save="model.pkl"
)

# 禁用自动转换（必须明确指定列名）
poisson_train(
    csv="data.csv",
    value_column="count",  # 必须指定
    autoconvert=False,     # 不自动转换
    save="model.pkl"
)
```

### CSV格式（传统方式）

CSV文件包含整数计数列，可选包含时间戳列：

```csv
timestamp,value
2024-01-01 00:00:00,2
2024-01-01 01:00:00,1
2024-01-01 02:00:00,8
...
```

**注意：** 数值必须是非负整数（计数数据）。

### TXT格式

TXT文件每行为空格分隔的整数计数值：

```
2 1 8 3 2
1 0 2 15 1
...
```

## 工作原理

### 泊松分布介绍

泊松分布只有一个参数：

- **λ（lambda）**：均值，同时也是方差

概率质量函数：

```
P(X = k) = (λ^k * e^(-λ)) / k!
```

分布特性：

- **均值**：E(X) = λ
- **方差**：Var(X) = λ
- **适用条件**：方差≈均值

### 参数估计方法

泊松分布的参数估计非常简单：

λ = 样本均值

这就是最大似然估计的结果。

**适用条件**：方差/均值 ≈ 1

如果 方差/均值 > 2，说明数据呈现过离散特征，应考虑使用负二项分布。

### 异常检测方法

1. **训练阶段**：

   - 加载训练数据（非负整数计数值）
   - 计算样本均值和方差
   - 验证方差/均值比率（应≈1）
   - 设置λ = 均值
   - 根据 `threshold_percentile`计算异常阈值
2. **检测阶段**：

   - 计算每个数据点的累积概率 CDF(x)
   - 如果 CDF(x) < threshold_percentile 或 CDF(x) > (1 - threshold_percentile)，标记为异常
   - 计算异常分数（负对数似然）

### 异常分数计算

异常分数 = -log(PMF(x) + ε)

其中 PMF(x) 是概率质量函数值，ε 是避免log(0)的小常数。分数越高，表示越异常。

## 参数调优

### window_size（窗口大小）

- **较小值（如20-50）**：适用于快速变化的计数数据，模型更敏感
- **较大值（如100-200）**：适用于平稳的计数数据，模型更稳定
- **建议**：根据数据特性调整，通常50是一个好的起点

### threshold_percentile（阈值百分位数）

- **较小值（如0.001-0.01）**：更严格，只标记极端异常
- **较大值（如0.05-0.1）**：更宽松，标记更多可疑点
- **建议**：
  - 故障检测：使用0.01（1%）
  - 预警系统：使用0.05（5%）
  - 质量控制：使用0.02-0.03（2-3%）

## 使用示例

### 完整工作流程

```python
# 1. 训练模型
train_result = poisson_train(
    csv="failure_counts.csv",
    value_column="failure_count",
    window_size=50,
    threshold_percentile=0.01,
    save="failure_model.pkl"
)

# 2. 检查训练进度
task_status = get_task(task_id=train_result["task_id"])
print(f"训练进度: {task_status['progress'] * 100}%")

# 3. 训练完成后，进行异常检测
detect_result = poisson_detect(
    model_path="failure_model.pkl",
    csv="test_failures.csv",
    value_column="failure_count",
    save_result="anomaly_results.csv"
)

# 4. 获取检测结果
detection = get_task(task_id=detect_result["task_id"])
if detection["status"] == "succeeded":
    details = detection["result"]["detection_details"]
    print(f"检测到 {details['anomaly_count']} 个异常点")
    print(f"异常率: {details['anomaly_rate'] * 100:.2f}%")
    print(f"λ参数: {details['model_params']['lambda']:.2f}")
    print(f"方差/均值比: {details['model_params']['variance_mean_ratio']:.2f}")
    print(f"异常索引: {details['anomaly_indices']}")
```

## 环境变量配置

- `POISSON_MAX_CONCURRENT`: 最大并发任务数，默认 `4`

```bash
# Linux/Mac
export POISSON_MAX_CONCURRENT=8

# Windows
set POISSON_MAX_CONCURRENT=8
```

## 注意事项

1. **数据要求**：

   - 必须是非负整数（计数数据）
   - 建议方差/均值比 ≤ 2（接近1最佳）
   - 建议训练数据量 ≥ window_size
2. **分布适用性验证**：

   - 如果训练时提示"数据呈现过离散特征"，说明方差远大于均值
   - 此时应考虑使用负二项分布而非泊松分布
3. **性能优化**：

   - 对于大数据集，使用 `nrows`参数限制加载量
   - 检测时使用 `limit`参数限制返回数据量
   - 通过环境变量调整并发任务数
4. **模型保存**：

   - 模型使用pickle序列化
   - 确保保存路径有写权限
   - 模型文件包含所有必要参数

## 泊松分布 vs 负二项分布 vs Gamma分布

| 特性           | 泊松分布         | 负二项分布            | Gamma分布     |
| -------------- | ---------------- | --------------------- | ------------- |
| 数据类型       | 非负整数（计数） | 非负整数（计数）      | 非负连续      |
| 方差与均值关系 | 方差 = 均值      | 方差 > 均值（过离散） | 方差 > 均值   |
| 参数数量       | 1个（λ）        | 2个（n, p）           | 2个（α, β） |
| 适用场景       | 低频随机事件     | 高频波动事件          | 连续时长数据  |
| 典型应用       | 故障次数         | 订单量、访问量        | 设备运行时长  |

**选择建议**：

1. 计算数据的 方差/均值比率
2. 如果 比率 ≈ 1，使用**泊松分布**（本工具）
3. 如果 比率 > 1.5，使用**负二项分布**
4. 如果数据是连续值，使用**Gamma分布**

## 实际应用案例

### 案例1：服务器故障次数监控

**场景**：监控服务器每小时故障次数，检测异常故障高发时段。

**特点**：

- 故障是低频随机事件
- 故障次数通常较小（0, 1, 2, ...）
- 方差≈均值

**使用方法**：

```python
poisson_train(csv="server_failures.csv", value_column="failure_count", 
              threshold_percentile=0.01, save="failure_model.pkl")
```

### 案例2：API错误日志计数

**场景**：监控API每分钟错误日志数，检测系统异常。

**特点**：

- 正常情况下错误数很少
- 系统故障时错误数暴增
- 符合泊松分布特征

**使用方法**：

```python
poisson_train(csv="api_errors.csv", value_column="error_count",
              threshold_percentile=0.05, save="api_model.pkl")
```

### 案例3：客服投诉次数异常

**场景**：监控每天客服投诉次数，检测产品质量问题。

**特点**：

- 投诉是随机事件
- 正常情况下次数稳定
- 产品问题时投诉增多

**使用方法**：

```python
poisson_train(csv="complaints.csv", value_column="complaint_count",
              threshold_percentile=0.02, save="complaint_model.pkl")
```

## 技术栈

- **Python 3.11+**
- **numpy**: 数值计算
- **pandas**: 数据处理
- **scipy**: 泊松分布拟合和统计计算
- **fastmcp**: MCP服务器框架
