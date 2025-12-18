from typing import Optional, Dict, Any, List
import os
import threading
import uuid
import datetime
import traceback

from fastmcp import FastMCP
from poisson_detector import PoissonDetector, DataLoader


# 初始化MCP服务器
mcp = FastMCP("poisson-anomaly-detection", debug=True, log_level="DEBUG")

# 后台任务管理基础设施
# 内存任务注册表
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

# 并发控制（通过环境变量 POISSON_MAX_CONCURRENT 配置）
MAX_CONCURRENT = int(os.getenv("POISSON_MAX_CONCURRENT", "4"))
TASKS_SEM = threading.Semaphore(MAX_CONCURRENT)

# 中国标准时间 (UTC+08:00)
TZ_CN = datetime.timezone(datetime.timedelta(hours=8))


def _now_iso() -> str:
    """返回当前时间的ISO格式字符串（中国时区）"""
    return datetime.datetime.now(TZ_CN).isoformat()


def _create_task(task_type: str, params: Dict[str, Any]) -> str:
    """创建新任务并返回任务ID"""
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": task_type,
        "params": params,
        "status": "queued",  # queued|running|succeeded|failed|cancelled
        "progress": 0.0,
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "traceback": None,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
    return task_id


def _set_task(task_id: str, **updates):
    """更新任务状态"""
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(**updates)


def _get_task(task_id: str) -> Dict[str, Any]:
    """获取任务详情"""
    with TASKS_LOCK:
        return dict(TASKS.get(task_id, {}))


def _list_tasks() -> List[Dict[str, Any]]:
    """列出所有任务"""
    with TASKS_LOCK:
        return [dict(t) for t in TASKS.values()]


def _start_background(target, *args):
    """启动后台线程"""
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


def _load_data(csv: Optional[str], txt: Optional[str], 
               value_column: Optional[str], nrows: Optional[int],
               autoconvert: bool = True, time_window: str = "1min"):
    """加载数据（CSV或TXT）"""
    if (csv and txt) or (not csv and not txt):
        raise ValueError("请指定且仅指定 'csv' 或 'txt' 之一")
    
    if csv:
        data, timestamps = DataLoader.load_csv(
            csv, 
            value_column=value_column, 
            nrows=nrows,
            autoconvert=autoconvert,
            time_window=time_window
        )
    else:
        data = DataLoader.load_txt(txt, nrows=nrows)
        timestamps = None
    
    return data, timestamps


def _train_worker(task_id: str, params: Dict[str, Any]):
    """训练任务工作线程"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            # 加载训练数据
            data, _ = _load_data(
                params.get("csv"),
                params.get("txt"),
                params.get("value_column"),
                params.get("nrows"),
                params.get("autoconvert", True),
                params.get("time_window", "1min")
            )
            _set_task(task_id, progress=0.3)
            
            # 创建并训练模型
            detector = PoissonDetector(
                window_size=params.get("window_size", 50),
                threshold_percentile=params.get("threshold_percentile", 0.01)
            )
            _set_task(task_id, progress=0.5)
            
            detector.fit(data)
            _set_task(task_id, progress=0.8)
            
            # 保存模型（如果指定）
            if params.get("save"):
                detector.save(params.get("save"))
            
            result = {
                "status": "ok",
                "samples": int(len(data)),
                "window_size": int(detector.window_size),
                "threshold_percentile": float(detector.threshold_percentile),
                "model_params": {
                    "lambda": float(detector.lambda_),
                    "mean": float(detector.mean),
                    "variance": float(detector.var),
                    "variance_mean_ratio": float(detector.var / detector.mean) if detector.mean > 0 else 0,
                    "threshold_low": float(detector.threshold_low),
                    "threshold_high": float(detector.threshold_high)
                },
                "saved_to": params.get("save") or ""
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, 
                     completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(),
                 error=str(ex), traceback=traceback.format_exc())


def _detect_worker(task_id: str, params: Dict[str, Any]):
    """异常检测任务工作线程"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            # 加载模型
            detector = PoissonDetector.load(params["model_path"])
            _set_task(task_id, progress=0.3)
            
            # 加载测试数据
            data, timestamps = _load_data(
                params.get("csv"),
                params.get("txt"),
                params.get("value_column"),
                params.get("nrows"),
                params.get("autoconvert", True),
                params.get("time_window", "1min")
            )
            _set_task(task_id, progress=0.5)
            
            # 执行异常检测
            details = detector.detect_with_details(data)
            _set_task(task_id, progress=0.9)
            
            # 保存结果（如果指定）
            if params.get("save_result"):
                predictions = detector.predict(data)
                scores = detector.score(data)
                DataLoader.save_csv(
                    params.get("save_result"),
                    data,
                    predictions,
                    scores,
                    timestamps
                )
            
            # 限制返回的数据量
            limit = params.get("limit", 1000)
            if limit and limit < details["total_points"]:
                details["predictions"] = details["predictions"][:limit]
                details["scores"] = details["scores"][:limit]
                details["cdf_values"] = details["cdf_values"][:limit]
                details["returned_points"] = limit
            else:
                details["returned_points"] = details["total_points"]
            
            result = {
                "status": "ok",
                "detection_details": details,
                "saved_to": params.get("save_result") or ""
            }
            
            _set_task(task_id, status="succeeded", progress=1.0,
                     completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(),
                 error=str(ex), traceback=traceback.format_exc())


# ============ MCP 工具定义 ============

@mcp.tool()
def poisson_train(
    csv: Optional[str] = None,
    txt: Optional[str] = None,
    value_column: Optional[str] = None,
    window_size: int = 50,
    threshold_percentile: float = 0.01,
    nrows: Optional[int] = None,
    save: Optional[str] = None,
    autoconvert: bool = True,
    time_window: str = "1min",
) -> Dict[str, Any]:
    """
    创建泊松分布模型训练任务（异步），立即返回 task_id
    
    参数:
        csv: Optional[str] = None
            训练数据CSV文件路径
            csv 和 txt 必须二选一
        txt: Optional[str] = None
            训练数据TXT文件路径，每行为空格分隔的整数计数值
            csv 和 txt 必须二选一
        value_column: Optional[str] = None
            CSV文件中的数值列名（None时自动识别）
            支持自动识别包含关键词的列：value, count, num, amount, quantity等
        window_size: int = 50
            滑动窗口大小，用于拟合泊松分布参数
        threshold_percentile: float = 0.01
            异常阈值百分位数（0-1之间），低于此概率的点被标记为异常
            默认0.01表示1%的尾部概率
        nrows: Optional[int] = None
            仅加载前n行数据（可选，用于快速测试）
        save: Optional[str] = None
            如果提供，训练完成后保存模型到此路径
        autoconvert: bool = True
            是否自动转换数据格式。支持：
            1. 自动识别时间列和数值列
            2. 日志格式自动聚合为计数时序
            3. 自定义列名自动映射
        time_window: str = "1min"
            日志数据的时间聚合窗口（如 '1min', '5min', '1H', '1D'）
            仅在处理日志格式数据时有效
    
    返回:
        dict: {"status": "queued", "task_id": str, "type": "train"}
    
    说明:
        泊松分布适用于计数数据（方差≈均值），如故障次数、事件发生次数等。
        如果数据呈现过离散（方差>均值），建议使用负二项分布。
        
        支持的数据格式：
        1. 标准格式：包含 'time' 和 'value' 列
        2. 自定义列名：自动识别时间列和数值列
        3. 日志格式：自动聚合为计数时序
    
    示例:
        # 标准格式
        poisson_train(csv="data.csv", save="model.pkl")
        
        # 自定义列名（自动识别）
        poisson_train(csv="orders.csv", autoconvert=True, save="model.pkl")
        
        # 日志格式（自动聚合）
        poisson_train(csv="logs.csv", time_window="5min", save="model.pkl")
    """
    params = {
        "csv": csv,
        "txt": txt,
        "value_column": value_column,
        "window_size": window_size,
        "threshold_percentile": threshold_percentile,
        "nrows": nrows,
        "save": save,
        "autoconvert": autoconvert,
        "time_window": time_window,
    }
    
    task_id = _create_task("train", params)
    
    # 预验证输入
    if (csv and txt) or (not csv and not txt):
        msg = "请指定且仅指定 'csv' 或 'txt' 之一"
        _set_task(task_id, status="failed", error=msg, completed_at=_now_iso())
        return {"status": "failed", "task_id": task_id, "type": "train", "error": msg}
    
    _start_background(_train_worker, task_id, params)
    return {"status": "queued", "task_id": task_id, "type": "train"}


@mcp.tool()
def poisson_detect(
    model_path: str,
    csv: Optional[str] = None,
    txt: Optional[str] = None,
    value_column: Optional[str] = None,
    nrows: Optional[int] = None,
    limit: Optional[int] = 1000,
    save_result: Optional[str] = None,
    autoconvert: bool = True,
    time_window: str = "1min",
) -> Dict[str, Any]:
    """
    创建异常检测任务（异步），使用已训练的模型检测异常，立即返回 task_id
    
    参数:
        model_path: str
            已保存的模型文件路径（必需）
        csv: Optional[str] = None
            测试数据CSV文件路径
            csv 和 txt 必须二选一
        txt: Optional[str] = None
            测试数据TXT文件路径，每行为空格分隔的整数计数值
            csv 和 txt 必须二选一
        value_column: Optional[str] = None
            CSV文件中的数值列名（None时自动识别）
            支持自动识别包含关键词的列：value, count, num, amount, quantity等
        nrows: Optional[int] = None
            仅加载前n行数据（可选，用于快速测试）
        limit: Optional[int] = 1000
            返回的数据点数量限制，避免返回体过大
        save_result: Optional[str] = None
            如果提供，将检测结果保存到此CSV文件
        autoconvert: bool = True
            是否自动转换数据格式。支持：
            1. 自动识别时间列和数值列
            2. 日志格式自动聚合为计数时序
            3. 自定义列名自动映射
        time_window: str = "1min"
            日志数据的时间聚合窗口（如 '1min', '5min', '1H', '1D'）
            仅在处理日志格式数据时有效
    
    返回:
        dict: {"status": "queued", "task_id": str, "type": "detect"}
        任务完成后，result中包含:
            - total_points: 总数据点数
            - anomaly_count: 异常点数量
            - anomaly_rate: 异常率
            - anomaly_indices: 异常点索引列表
            - predictions: 异常标签（0/1）
            - scores: 异常分数
            - model_params: 模型参数（lambda, mean, variance等）
    
    示例:
        # 标准格式
        poisson_detect(model_path="model.pkl", csv="test.csv", save_result="result.csv")
        
        # 自定义列名（自动识别）
        poisson_detect(model_path="model.pkl", csv="orders.csv", autoconvert=True)
        
        # 日志格式（自动聚合）
        poisson_detect(model_path="model.pkl", csv="logs.csv", time_window="5min")
    """
    params = {
        "model_path": model_path,
        "csv": csv,
        "txt": txt,
        "value_column": value_column,
        "nrows": nrows,
        "limit": limit,
        "save_result": save_result,
        "autoconvert": autoconvert,
        "time_window": time_window,
    }
    
    task_id = _create_task("detect", params)
    
    # 预验证输入
    if (csv and txt) or (not csv and not txt):
        msg = "请指定且仅指定 'csv' 或 'txt' 之一"
        _set_task(task_id, status="failed", error=msg, completed_at=_now_iso())
        return {"status": "failed", "task_id": task_id, "type": "detect", "error": msg}
    
    if not model_path:
        msg = "参数 'model_path' (模型路径) 是必需的"
        _set_task(task_id, status="failed", error=msg, completed_at=_now_iso())
        return {"status": "failed", "task_id": task_id, "type": "detect", "error": msg}
    
    if not os.path.exists(model_path):
        msg = f"模型文件不存在: {model_path}"
        _set_task(task_id, status="failed", error=msg, completed_at=_now_iso())
        return {"status": "failed", "task_id": task_id, "type": "detect", "error": msg}
    
    _start_background(_detect_worker, task_id, params)
    return {"status": "queued", "task_id": task_id, "type": "detect"}


@mcp.tool()
def list_tasks() -> Dict[str, Any]:
    """
    列出当前服务器跟踪的所有后台任务
    
    返回:
        dict: {
            "count": int, 任务数量
            "tasks": list, 任务对象列表
        }
        每个任务对象包含: id, type, params, status, progress, 时间戳, result/error等
    """
    tasks = _list_tasks()
    return {"count": len(tasks), "tasks": tasks}


@mcp.tool()
def get_task(task_id: str) -> Dict[str, Any]:
    """
    获取指定任务的详细状态、进度和结果
    
    参数:
        task_id: str
            创建任务时返回的任务ID（由 poisson_train 或 poisson_detect 返回）
    
    返回:
        dict: 完整的任务对象，如果task_id未知则返回空对象
        任务对象包含:
            - id: 任务ID
            - type: 任务类型 (train/detect)
            - params: 任务参数
            - status: 状态 (queued/running/succeeded/failed)
            - progress: 进度 (0.0-1.0)
            - created_at: 创建时间
            - started_at: 开始时间
            - completed_at: 完成时间
            - result: 结果（成功时）
            - error: 错误信息（失败时）
            - traceback: 错误堆栈（失败时）
    """
    return _get_task(task_id)


@mcp.tool()
def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    取消指定的任务（仅对排队中的任务有效）
    
    参数:
        task_id: str
            要取消的任务ID
    
    返回:
        dict: {"status": "ok"/"failed", "message": str}
    """
    task = _get_task(task_id)
    
    if not task:
        return {"status": "failed", "message": f"任务不存在: {task_id}"}
    
    if task["status"] == "queued":
        _set_task(task_id, status="cancelled", completed_at=_now_iso())
        return {"status": "ok", "message": f"任务已取消: {task_id}"}
    elif task["status"] == "running":
        return {"status": "failed", "message": "无法取消正在运行的任务"}
    else:
        return {"status": "failed", "message": f"任务已处于终止状态: {task['status']}"}


@mcp.tool()
def clear_tasks() -> Dict[str, Any]:
    """
    清除所有已完成（成功/失败/取消）的任务记录
    
    返回:
        dict: {"status": "ok", "cleared": int, "remaining": int}
    """
    with TASKS_LOCK:
        before_count = len(TASKS)
        TASKS.clear()
        cleared = before_count
        remaining = len(TASKS)
    
    return {"status": "ok", "cleared": cleared, "remaining": remaining}


if __name__ == "__main__":
    # 启动SSE传输的MCP服务器，默认端口2252
    mcp.run(transport="sse", port=2252)

