#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HeyGem AI 数字人生成器 - 激进GPU推理优化引擎
专为RTX 4090优化设计，实现极致推理性能

优化策略:
1. TensorRT引擎自动转换和缓存
2. FP16/INT8精度优化
3. CUDA内存池管理
4. 异步推理和批处理
5. CUDA Graphs加速
6. 多模型并行推理
"""

import os
import gc
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import torch
    import onnxruntime as ort
    from onnxruntime.capi import _pybind_state as C
    import cv2
    import psutil
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 依赖包缺失: {e}")
    DEPENDENCIES_AVAILABLE = False

from y_utils.logger import logger


class RTX4090OptimizationEngine:
    """RTX 4090 激进优化引擎"""

    def __init__(self):
        self.device_id = 0
        self.max_workspace_size = 8 * 1024 * 1024 * 1024  # 8GB workspace for TensorRT
        # 缓存目录配置（统一放在temp目录）
        self.engine_cache_dir = Path("./temp/tensorrt_cache")
        self.engine_cache_dir.mkdir(parents=True, exist_ok=True)

        # 性能监控
        self.performance_stats = {
            'inference_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }

        # 模型缓存
        self.model_cache = {}
        self.session_cache = {}

        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # 初始化优化器
        self._initialize_optimization()

    def _initialize_optimization(self):
        """初始化GPU优化配置"""
        logger.info("🚀 初始化RTX 4090激进优化引擎...")

        if not DEPENDENCIES_AVAILABLE:
            logger.error("❌ 必要依赖包未安装，无法进行优化")
            return

        # 1. CUDA内存优化
        self._optimize_cuda_memory()

        # 2. PyTorch优化
        self._optimize_pytorch()

        # 3. ONNX Runtime优化
        self._optimize_onnx_runtime()

        # 4. 系统级优化
        self._optimize_system()

        logger.info("✅ RTX 4090优化引擎初始化完成")

    def _optimize_cuda_memory(self):
        """优化CUDA内存管理"""
        logger.info("💾 配置CUDA内存优化...")

        if torch.cuda.is_available():
            # 设置内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'

            # 清空CUDA缓存
            torch.cuda.empty_cache()

            # 设置CUDA内存增长策略
            torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%显存

            # 预分配内存池
            try:
                # 预热GPU内存
                dummy_tensor = torch.randn(1024, 1024, device='cuda')
                del dummy_tensor
                torch.cuda.empty_cache()
                logger.info("✅ CUDA内存预热完成")
            except Exception as e:
                logger.warning(f"⚠️ CUDA内存预热失败: {e}")

    def _optimize_pytorch(self):
        """优化PyTorch性能"""
        logger.info("🔥 配置PyTorch优化...")

        if torch.cuda.is_available():
            # 启用TensorFloat-32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # 启用cudnn benchmark
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # 启用融合优化
            torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])

            logger.info("✅ PyTorch优化配置完成")

    def _optimize_onnx_runtime(self):
        """优化ONNX Runtime配置"""
        logger.info("⚡ 配置ONNX Runtime优化...")

        # 获取可用提供者
        available_providers = ort.get_available_providers()
        logger.info(f"可用执行提供者: {available_providers}")

        # 检查TensorRT支持
        self.tensorrt_available = 'TensorrtExecutionProvider' in available_providers
        logger.info(f"TensorRT支持: {self.tensorrt_available}")

        if self.tensorrt_available:
            logger.info("🎯 TensorRT可用，将启用激进优化模式")
        else:
            logger.warning("⚠️ TensorRT不可用，使用CUDA优化模式")

    def _optimize_system(self):
        """系统级性能优化"""
        logger.info("🔧 配置系统级优化...")

        # 设置进程优先级
        try:
            import psutil
            process = psutil.Process()
            if os.name == 'posix':  # Linux/Unix
                process.nice(-10)  # 提高进程优先级
            logger.info("✅ 进程优先级已优化")
        except Exception as e:
            logger.warning(f"⚠️ 无法设置进程优先级: {e}")

        # CPU亲和性设置
        try:
            if hasattr(os, 'sched_setaffinity'):
                # 使用所有CPU核心
                cpu_count = os.cpu_count()
                os.sched_setaffinity(0, range(cpu_count))
                logger.info(f"✅ CPU亲和性已设置: {cpu_count}核心")
        except Exception as e:
            logger.warning(f"⚠️ 无法设置CPU亲和性: {e}")

    def get_optimal_providers(self, model_path: str, input_shapes: Dict) -> List[Tuple[str, Dict]]:
        """获取最优执行提供者配置"""
        providers = []

        if self.tensorrt_available:
            # TensorRT提供者配置
            trt_options = {
                'device_id': self.device_id,
                'trt_max_workspace_size': self.max_workspace_size,
                'trt_fp16_enable': True,  # 启用FP16
                'trt_int8_enable': False,  # 可选启用INT8
                'trt_int8_calibration_table_name': '',
                'trt_dla_enable': False,
                'trt_dla_core': 0,
                'trt_dump_subgraphs': False,
                'trt_engine_cache_enable': True,  # 启用引擎缓存
                'trt_engine_cache_path': str(self.engine_cache_dir),
                'trt_force_sequential_engine_build': False,
                'trt_context_memory_sharing_enable': True,
                'trt_layer_norm_fp32_fallback': False,
                'trt_timing_cache_enable': True,
                'trt_force_timing_cache': False,
                'trt_detailed_build_log': False,
                'trt_build_heuristics_enable': True,
                'trt_sparsity_enable': False,
                'trt_builder_optimization_level': 5,  # 最高优化级别
                'trt_auxiliary_streams': 1,
                'trt_tactic_sources': '+CUDNN,+CUBLAS,+CUBLAS_LT',
                'trt_extra_plugin_lib_paths': '',
                'trt_profile_min_shapes': '',
                'trt_profile_max_shapes': '',
                'trt_profile_opt_shapes': '',
                'trt_cuda_graph_enable': True,  # 启用CUDA图
            }

            # 动态形状配置
            if input_shapes:
                min_shapes = []
                max_shapes = []
                opt_shapes = []

                for name, shape in input_shapes.items():
                    if isinstance(shape, list) and len(shape) > 0:
                        # 假设第一个维度是batch size
                        min_shape = [1] + shape[1:] if len(shape) > 1 else [1]
                        max_shape = [16] + shape[1:] if len(shape) > 1 else [16]  # 支持到batch=16
                        opt_shape = [4] + shape[1:] if len(shape) > 1 else [4]   # 优化batch=4

                        min_shapes.append(f"{name}:{','.join(map(str, min_shape))}")
                        max_shapes.append(f"{name}:{','.join(map(str, max_shape))}")
                        opt_shapes.append(f"{name}:{','.join(map(str, opt_shape))}")

                if min_shapes:
                    trt_options['trt_profile_min_shapes'] = '|'.join(min_shapes)
                    trt_options['trt_profile_max_shapes'] = '|'.join(max_shapes)
                    trt_options['trt_profile_opt_shapes'] = '|'.join(opt_shapes)

            providers.append(('TensorrtExecutionProvider', trt_options))

        # CUDA提供者配置
        cuda_options = {
            'device_id': self.device_id,
            'arena_extend_strategy': 'kSameAsRequested',  # 内存分配策略
            'gpu_mem_limit': int(20 * 1024 * 1024 * 1024),  # 20GB显存限制
            'cudnn_conv_algo_search': 'HEURISTIC',  # 卷积算法搜索
            'do_copy_in_default_stream': True,  # 默认流复制
            'cudnn_conv_use_max_workspace': True,  # 使用最大工作空间
            'cudnn_conv1d_pad_to_nc1d': True,  # 1D卷积优化
            'enable_cuda_graph': True,  # 启用CUDA图
            'cuda_graph_enable_building': True,
            'tunable_op_enable': True,  # 启用可调优算子
            'tunable_op_tuning_enable': True,
        }
        providers.append(('CUDAExecutionProvider', cuda_options))

        # CPU后备
        providers.append(('CPUExecutionProvider', {
            'intra_op_num_threads': 0,  # 使用所有CPU线程
            'inter_op_num_threads': 0,
        }))

        return providers

    def get_optimal_session_options(self) -> ort.SessionOptions:
        """获取最优会话选项"""
        session_options = ort.SessionOptions()

        # 基本配置
        session_options.intra_op_num_threads = 0  # 使用所有线程
        session_options.inter_op_num_threads = 0
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 高级优化
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.enable_profiling = False  # 生产环境关闭profiling

        # 日志级别
        session_options.log_severity_level = 3  # 只显示错误
        session_options.log_verbosity_level = 0

        return session_options

    def create_optimized_session(self, model_path: str, input_shapes: Dict = None) -> ort.InferenceSession:
        """创建优化的推理会话"""
        # 检查缓存
        cache_key = f"{model_path}_{hash(str(input_shapes))}"
        if cache_key in self.session_cache:
            return self.session_cache[cache_key]

        logger.info(f"🎯 创建优化推理会话: {model_path}")

        # 获取最优配置
        providers = self.get_optimal_providers(model_path, input_shapes or {})
        session_options = self.get_optimal_session_options()

        try:
            # 创建会话
            session = ort.InferenceSession(
                model_path,
                session_options,
                providers=providers
            )

            # 验证提供者
            active_providers = session.get_providers()
            logger.info(f"激活的执行提供者: {active_providers}")

            # 缓存会话
            self.session_cache[cache_key] = session

            return session

        except Exception as e:
            logger.error(f"❌ 创建优化会话失败: {e}")
            # 降级到基本CUDA配置
            basic_providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
            session = ort.InferenceSession(model_path, session_options, providers=basic_providers)
            self.session_cache[cache_key] = session
            return session

    def benchmark_model(self, session: ort.InferenceSession, input_data: Dict, iterations: int = 100) -> Dict:
        """基准测试模型性能"""
        logger.info(f"📊 开始性能基准测试 ({iterations}次迭代)...")

        # 预热
        for _ in range(10):
            try:
                session.run(None, input_data)
            except Exception as e:
                logger.warning(f"预热失败: {e}")
                break

        # 清空GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 基准测试
        start_time = time.time()
        inference_times = []

        for i in range(iterations):
            iter_start = time.time()
            try:
                outputs = session.run(None, input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # 等待GPU完成
                iter_time = time.time() - iter_start
                inference_times.append(iter_time)
            except Exception as e:
                logger.error(f"推理失败 (iteration {i}): {e}")
                break

        total_time = time.time() - start_time

        if inference_times:
            stats = {
                'total_time': total_time,
                'avg_inference_time': np.mean(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'std_inference_time': np.std(inference_times),
                'throughput_fps': iterations / total_time,
                'iterations': len(inference_times),
                'success_rate': len(inference_times) / iterations
            }

            logger.info("📈 性能基准测试结果:")
            logger.info(f"  平均推理时间: {stats['avg_inference_time']*1000:.2f}ms")
            logger.info(f"  最小推理时间: {stats['min_inference_time']*1000:.2f}ms")
            logger.info(f"  最大推理时间: {stats['max_inference_time']*1000:.2f}ms")
            logger.info(f"  吞吐量: {stats['throughput_fps']:.2f} FPS")
            logger.info(f"  成功率: {stats['success_rate']*100:.1f}%")

            return stats
        else:
            return {'error': 'All iterations failed'}

    def optimize_existing_models(self, model_dir: str = "."):
        """批量优化现有模型"""
        logger.info("🔥 开始批量优化现有ONNX模型...")

        # 查找所有ONNX模型
        onnx_files = list(Path(model_dir).rglob("*.onnx"))

        optimization_results = []

        for model_path in onnx_files:
            logger.info(f"🎯 优化模型: {model_path}")

            try:
                # 分析模型
                model_info = self.analyze_model(str(model_path))

                # 创建优化会话
                session = self.create_optimized_session(str(model_path))

                # 生成测试数据
                input_data = self.generate_test_input(session)

                if input_data:
                    # 性能测试
                    perf_stats = self.benchmark_model(session, input_data, iterations=50)

                    result = {
                        'model_path': str(model_path),
                        'model_info': model_info,
                        'performance': perf_stats,
                        'providers': session.get_providers()
                    }
                    optimization_results.append(result)

                    logger.info(f"✅ 模型 {model_path.name} 优化完成")
                else:
                    logger.warning(f"⚠️ 无法为模型 {model_path.name} 生成测试数据")

            except Exception as e:
                logger.error(f"❌ 优化模型 {model_path} 失败: {e}")

        # 保存优化报告
        report_path = Path("gpu_optimization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📊 优化报告已保存: {report_path}")
        return optimization_results

    def analyze_model(self, model_path: str) -> Dict:
        """分析模型结构和特征"""
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

            inputs_info = []
            for inp in session.get_inputs():
                inputs_info.append({
                    'name': inp.name,
                    'type': str(inp.type),
                    'shape': inp.shape
                })

            outputs_info = []
            for out in session.get_outputs():
                outputs_info.append({
                    'name': out.name,
                    'type': str(out.type),
                    'shape': out.shape
                })

            # 估算模型大小
            model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB

            return {
                'inputs': inputs_info,
                'outputs': outputs_info,
                'model_size_mb': model_size,
                'input_count': len(inputs_info),
                'output_count': len(outputs_info)
            }

        except Exception as e:
            logger.error(f"分析模型失败 {model_path}: {e}")
            return {'error': str(e)}

    def generate_test_input(self, session: ort.InferenceSession) -> Optional[Dict]:
        """为模型生成测试输入数据"""
        try:
            input_data = {}

            for inp in session.get_inputs():
                shape = inp.shape
                # 处理动态维度
                processed_shape = []
                for dim in shape:
                    if isinstance(dim, str) or dim < 0:
                        # 动态维度，使用合理的默认值
                        if len(processed_shape) == 0:  # batch dimension
                            processed_shape.append(1)
                        else:
                            processed_shape.append(640)  # 常见的输入尺寸
                    else:
                        processed_shape.append(dim)

                # 生成随机数据
                if 'float' in str(inp.type).lower():
                    data = np.random.randn(*processed_shape).astype(np.float32)
                elif 'int' in str(inp.type).lower():
                    data = np.random.randint(0, 256, processed_shape, dtype=np.int32)
                else:
                    data = np.random.randn(*processed_shape).astype(np.float32)

                input_data[inp.name] = data

            return input_data

        except Exception as e:
            logger.error(f"生成测试输入失败: {e}")
            return None

    def clear_cache(self):
        """清理缓存"""
        self.session_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 缓存已清理")

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        return {
            'stats': self.performance_stats,
            'cache_size': len(self.session_cache),
            'tensorrt_available': self.tensorrt_available,
            'device_info': self._get_device_info()
        }

    def _get_device_info(self) -> Dict:
        """获取设备信息"""
        info = {}

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_name': props.name,
                'gpu_memory_total': props.total_memory / 1024**3,
                'gpu_memory_free': (props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            })

        info.update({
            'cpu_count': os.cpu_count(),
            'system_memory': psutil.virtual_memory().total / 1024**3
        })

        return info


# 全局优化引擎实例
_optimization_engine = None

def get_optimization_engine() -> RTX4090OptimizationEngine:
    """获取全局优化引擎实例"""
    global _optimization_engine
    if _optimization_engine is None:
        _optimization_engine = RTX4090OptimizationEngine()
    return _optimization_engine


if __name__ == "__main__":
    # 测试优化引擎
    engine = RTX4090OptimizationEngine()

    # 批量优化现有模型
    results = engine.optimize_existing_models()

    # 显示性能报告
    report = engine.get_performance_report()
    print(f"📊 性能报告: {json.dumps(report, indent=2, ensure_ascii=False, default=str)}")