#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 优化的ONNX模型包装器
专为RTX 4090和数字人生成优化
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

try:
    import onnxruntime as ort
    import torch
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

from y_utils.logger import logger


class OptimizedONNXModel:
    """优化的ONNX模型包装器"""

    def __init__(self, model_path: str, provider: str = 'auto', input_dynamic_shape: Dict = None):
        self.model_path = model_path
        self.input_dynamic_shape = input_dynamic_shape
        self.session = None
        self.input_names = []
        self.output_names = []

        # 性能统计
        self.inference_count = 0
        self.total_time = 0.0
        self.warmup_done = False

        # 初始化会话
        self._initialize_session(provider)

    def _initialize_session(self, provider: str):
        """初始化优化的推理会话"""
        if not HAS_DEPENDENCIES:
            raise RuntimeError("❌ 缺少必要依赖: onnxruntime, torch")

        logger.info(f"🎯 初始化优化ONNX模型: {Path(self.model_path).name}")

        # 获取最优提供者配置
        providers = self._get_optimal_providers(provider)

        # 会话选项
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 0  # 使用所有线程
        session_options.inter_op_num_threads = 0
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.log_severity_level = 3  # 只显示错误

        try:
            # 创建会话
            self.session = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=providers
            )

            # 获取输入输出信息
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]

            # 记录使用的提供者
            active_providers = self.session.get_providers()
            logger.info(f"✅ 模型加载成功，使用提供者: {active_providers[0]}")

            # 预热
            self._warmup()

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            # 降级到基本CUDA配置
            fallback_providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, session_options, providers=fallback_providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info("⚠️ 使用降级配置")

    def _get_optimal_providers(self, provider: str) -> List:
        """获取最优执行提供者"""
        available_providers = ort.get_available_providers()

        if provider == 'auto':
            # 自动选择最优提供者
            if 'TensorrtExecutionProvider' in available_providers:
                return self._get_tensorrt_providers()
            elif 'CUDAExecutionProvider' in available_providers:
                return self._get_cuda_providers()
            else:
                return ['CPUExecutionProvider']
        elif provider == 'tensorrt' and 'TensorrtExecutionProvider' in available_providers:
            return self._get_tensorrt_providers()
        elif provider == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            return self._get_cuda_providers()
        else:
            return ['CPUExecutionProvider']

    def _get_tensorrt_providers(self) -> List:
        """获取TensorRT提供者配置"""
        # 缓存目录配置（统一放在temp目录）
        cache_dir = Path("./temp/tensorrt_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        trt_options = {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(cache_dir),
            'trt_context_memory_sharing_enable': True,
            'trt_timing_cache_enable': True,
            'trt_builder_optimization_level': 5,
            'trt_auxiliary_streams': 1,
            'trt_cuda_graph_enable': True,
        }

        return [
            ('TensorrtExecutionProvider', trt_options),
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider'
        ]

    def _get_cuda_providers(self) -> List:
        """获取CUDA提供者配置"""
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': int(16 * 1024 * 1024 * 1024),  # 16GB
            'cudnn_conv_algo_search': 'HEURISTIC',
            'do_copy_in_default_stream': True,
            'cudnn_conv_use_max_workspace': True,
            'enable_cuda_graph': True,
            'tunable_op_enable': True,
            'tunable_op_tuning_enable': True,
        }

        return [
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider'
        ]

    def _warmup(self, warmup_iterations: int = 5):
        """预热模型"""
        if self.warmup_done:
            return

        logger.info(f"🔥 预热模型...")

        try:
            # 生成测试输入
            input_data = self._generate_test_input()

            if input_data:
                # 预热推理
                for _ in range(warmup_iterations):
                    self.session.run(None, input_data)

                # 同步GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                self.warmup_done = True
                logger.info("✅ 模型预热完成")
            else:
                logger.warning("⚠️ 无法生成测试数据，跳过预热")

        except Exception as e:
            logger.warning(f"⚠️ 预热失败: {e}")

    def _generate_test_input(self) -> Optional[Dict]:
        """生成测试输入数据"""
        try:
            input_data = {}

            for inp in self.session.get_inputs():
                shape = inp.shape
                # 处理动态维度
                processed_shape = []
                for i, dim in enumerate(shape):
                    if isinstance(dim, str) or dim < 0:
                        if i == 0:  # batch dimension
                            processed_shape.append(1)
                        elif i in [2, 3]:  # 常见的H,W维度
                            processed_shape.append(640)
                        else:
                            processed_shape.append(64)
                    else:
                        processed_shape.append(dim)

                # 生成随机数据
                if 'float' in str(inp.type).lower():
                    data = np.random.randn(*processed_shape).astype(np.float32)
                    # 如果是图像数据，归一化到合理范围
                    if len(processed_shape) == 4 and processed_shape[1] == 3:  # NCHW图像
                        data = (data * 0.5 + 0.5) * 255  # [0, 255]
                else:
                    data = np.random.randint(0, 256, processed_shape).astype(np.float32)

                input_data[inp.name] = data

            return input_data

        except Exception as e:
            logger.error(f"生成测试输入失败: {e}")
            return None

    def __call__(self, *args, **kwargs):
        """推理调用"""
        return self.run(*args, **kwargs)

    def run(self, input_data: Dict) -> List:
        """执行推理"""
        start_time = time.time()

        try:
            # 预处理输入数据
            processed_input = self._preprocess_input(input_data)

            # 执行推理
            outputs = self.session.run(None, processed_input)

            # 后处理输出
            outputs = self._postprocess_output(outputs)

            # 统计性能
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_time += inference_time

            return outputs

        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            raise

    def _preprocess_input(self, input_data: Dict) -> Dict:
        """预处理输入数据"""
        processed = {}

        for name, data in input_data.items():
            if isinstance(data, np.ndarray):
                # 确保数据类型正确
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                processed[name] = data
            else:
                # 转换为numpy数组
                processed[name] = np.array(data, dtype=np.float32)

        return processed

    def _postprocess_output(self, outputs: List) -> List:
        """后处理输出数据"""
        # 可以在这里添加特定的后处理逻辑
        return outputs

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if self.inference_count > 0:
            avg_time = self.total_time / self.inference_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = 0
            fps = 0

        return {
            'inference_count': self.inference_count,
            'total_time': self.total_time,
            'average_time': avg_time,
            'fps': fps,
            'model_path': self.model_path,
            'providers': self.session.get_providers() if self.session else []
        }

    def benchmark(self, iterations: int = 100) -> Dict:
        """性能基准测试"""
        logger.info(f"📊 开始性能基准测试 ({iterations}次)...")

        # 生成测试数据
        test_input = self._generate_test_input()
        if not test_input:
            return {'error': '无法生成测试数据'}

        # 预热
        for _ in range(10):
            self.session.run(None, test_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 基准测试
        times = []
        start_total = time.time()

        for i in range(iterations):
            start = time.time()
            self.session.run(None, test_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)

        total_time = time.time() - start_total

        stats = {
            'iterations': iterations,
            'total_time': total_time,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'fps': iterations / total_time,
            'model_name': Path(self.model_path).name,
            'providers': self.session.get_providers()
        }

        logger.info(f"📈 基准测试完成:")
        logger.info(f"  平均时间: {stats['avg_time']*1000:.2f}ms")
        logger.info(f"  FPS: {stats['fps']:.2f}")

        return stats

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session') and self.session:
            del self.session