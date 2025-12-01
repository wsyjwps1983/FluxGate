from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
import os
from dotenv import load_dotenv

from semantic_router import Route, SemanticRouter, HybridRouter
from semantic_router.encoders import LocalEncoder, BM25Encoder, SiliconFlowEncoder
from semantic_router.index import FaissIndex
from semantic_router.utils.logger import logger

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="FlaxGate API",
    description="基于语义的意图识别API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 从环境变量获取配置
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
ALPHA = float(os.getenv("ALPHA", "0.3"))
ENCODER_TYPE = os.getenv("ENCODER_TYPE", "local")  # 支持: local, siliconflow
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")

# 初始化路由器
if ENCODER_TYPE == "siliconflow":
    if not SILICONFLOW_API_KEY:
        raise ValueError("使用SiliconFlow编码器时，必须设置SILICONFLOW_API_KEY环境变量")
    encoder = SiliconFlowEncoder(
        name=MODEL_NAME,
        siliconflow_api_key=SILICONFLOW_API_KEY,
        siliconflow_base_url=SILICONFLOW_BASE_URL
    )
else:
    encoder = LocalEncoder(name=MODEL_NAME)  # 使用环境变量配置模型

sparse_encoder = BM25Encoder()
index = FaissIndex()

# 初始化HybridRouter
router = HybridRouter(
    encoder=encoder,
    sparse_encoder=sparse_encoder,
    index=index,
    alpha=ALPHA
)

# 从文件加载初始意图数据
def load_initial_intents():
    """从final_training_data.json加载初始意图数据"""
    # 使用相对路径，从当前文件向上两层到项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 当前文件路径: semantic_router/api/main.py
    # 向上一层: semantic_router/
    # 向上两层: 项目根目录
    project_root = os.path.dirname(os.path.dirname(current_dir))
    file_path = os.path.join(project_root, "final_training_data.json")
    
    if not os.path.exists(file_path):
        logger.warning(f"初始意图数据文件不存在: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        # 按意图分组
        grouped_intents = {}
        for intent in intents:
            name = intent["intent"]
            question = intent["question"]
            if name not in grouped_intents:
                grouped_intents[name] = []
            grouped_intents[name].append(question)
        
        # 添加路由
        for intent_name, utterances in grouped_intents.items():
            route = Route(
                name=intent_name,
                utterances=utterances,
                score_threshold=0.7,
                description=f"{intent_name}意图路由",
                metadata={}
            )
            router.add(route)
        
        logger.info(f"成功加载 {len(grouped_intents)} 个意图路由，共 {len(intents)} 条样本")
    except Exception as e:
        logger.error(f"加载初始意图数据失败: {str(e)}")

# 加载初始意图数据
load_initial_intents()

# 数据模型
class PredictRequest(BaseModel):
    """预测请求模型"""
    query: str
    top_k: Optional[int] = 1

class PredictResponse(BaseModel):
    """预测响应模型"""
    query: str
    intent: Optional[str]
    matched: bool
    similarity_score: Optional[float]
    top_k_results: List[Dict[str, Any]]

class RouteRequest(BaseModel):
    """路由请求模型"""
    name: str
    utterances: List[str]
    score_threshold: Optional[float] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    message: str
    version: str

# API端点
@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="ok",
        message="Semantic Router API is running",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse, tags=["预测"])
def predict(request: PredictRequest):
    """预测单个问题的意图
    
    Args:
        request: 预测请求，包含查询文本和top_k参数
        
    Returns:
        预测响应，包含意图类型、是否命中、相似度得分和top_k结果
    """
    try:
        # 调用路由器进行预测
        result = router(request.query, limit=request.top_k)
        
        # 处理结果
        if isinstance(result, list):
            top_k_results = []
            for item in result:
                top_k_results.append({
                    "intent": item.name,
                    "similarity_score": item.similarity_score,
                    "function_call": item.function_call
                })
            
            # 确定最佳匹配
            best_result = top_k_results[0]
            intent = best_result["intent"]
            similarity_score = best_result["similarity_score"]
            matched = similarity_score is not None and similarity_score > 0
        else:
            # 单个结果
            top_k_results = [{
                "intent": result.name,
                "similarity_score": result.similarity_score,
                "function_call": result.function_call
            }]
            intent = result.name
            similarity_score = result.similarity_score
            matched = similarity_score is not None and similarity_score > 0
        
        return PredictResponse(
            query=request.query,
            intent=intent,
            matched=matched,
            similarity_score=similarity_score,
            top_k_results=top_k_results
        )
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.get("/routes", tags=["路由管理"])
def get_routes():
    """获取所有路由"""
    try:
        routes = []
        for route in router.routes:
            routes.append({
                "name": route.name,
                "utterances": route.utterances,
                "score_threshold": route.score_threshold,
                "description": route.description,
                "metadata": route.metadata
            })
        return {"routes": routes, "count": len(routes)}
    except Exception as e:
        logger.error(f"获取路由失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取路由失败: {str(e)}")

@app.post("/routes", tags=["路由管理"])
def add_route(request: RouteRequest):
    """添加新路由"""
    try:
        # 创建Route对象
        route = Route(
            name=request.name,
            utterances=request.utterances,
            score_threshold=request.score_threshold,
            description=request.description,
            metadata=request.metadata
        )
        
        # 添加到路由器
        router.add(route)
        
        return {
            "status": "success",
            "message": f"路由 {request.name} 添加成功",
            "route": {
                "name": route.name,
                "utterances": route.utterances,
                "score_threshold": route.score_threshold,
                "description": route.description,
                "metadata": route.metadata
            }
        }
    except Exception as e:
        logger.error(f"添加路由失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加路由失败: {str(e)}")

@app.put("/routes/{name}", tags=["路由管理"])
def update_route(name: str, request: RouteRequest):
    """更新路由"""
    try:
        # 检查路由是否存在
        existing_route = next((r for r in router.routes if r.name == name), None)
        if not existing_route:
            raise HTTPException(status_code=404, detail=f"路由 {name} 不存在")
        
        # 更新路由
        router.update(
            name=name,
            threshold=request.score_threshold,
            utterances=request.utterances
        )
        
        return {
            "status": "success",
            "message": f"路由 {name} 更新成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新路由失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新路由失败: {str(e)}")

@app.delete("/routes/{name}", tags=["路由管理"])
def delete_route(name: str):
    """删除路由"""
    try:
        # 删除路由
        router.delete(name)
        
        return {
            "status": "success",
            "message": f"路由 {name} 删除成功"
        }
    except Exception as e:
        logger.error(f"删除路由失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除路由失败: {str(e)}")

@app.get("/stats", tags=["统计信息"])
def get_stats():
    """获取统计信息"""
    try:
        stats = {
            "total_routes": len(router.routes),
            "total_utterances": sum(len(route.utterances) for route in router.routes),
            "index_type": router.index.type,
            "encoder_type": type(router.encoder).__name__,
            "sparse_encoder_type": type(router.sparse_encoder).__name__ if router.sparse_encoder else None,
            "alpha": router.alpha if hasattr(router, 'alpha') else None,
            "score_threshold": router.score_threshold
        }
        return stats
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

# 主函数
if __name__ == "__main__":
    uvicorn.run(
        "semantic_router.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )