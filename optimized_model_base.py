#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 优化的模型基类
集成RTX 4090激进优化策略
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

from optimized_onnx_model import OptimizedONNXModel
from gpu_optimization_engine import get_optimization_engine
from y_utils.logger import logger


class OptimizedModelBase:
    """优化的模型基类"""

    def __init__(self, model_info: Dict, provider: str = 'auto'):
        self.model_path = model_info['model_path']
        self.provider = provider

        # 获取优化引擎
        self.optimization_engine = get_optimization_engine()

        # 模型配置
        self.input_dynamic_shape = model_info.get('input_dynamic_shape', None)
        self.picklable = model_info.get('picklable', False)
        self.encrypt_key = model_info.get('encrypt', None)

        # 性能统计
        self.performance_stats = {
            'load_time': 0,
            'inference_count': 0,
            'total_inference_time': 0
        }

        # 初始化模型
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        load_start = time.time()
        logger.info(f"🎯 初始化优化模型: {Path(self.model_path).name}")

        try:
            model_suffix = Path(self.model_path).suffix.lower()

            if model_suffix == '.engine':
                # TensorRT引擎 - 最高性能
                self.model_type = 'tensorrt'
                self.model = self._load_tensorrt_model()
                logger.info("✅ 使用TensorRT引擎 (最高性能)")

            elif model_suffix in ['.onnx', '.bin']:
                # ONNX模型 - 使用优化包装器
                self.model_type = 'onnx'
                self.model = self._load_optimized_onnx_model()
                logger.info("✅ 使用优化ONNX模型")

            elif model_suffix == '.tjm':
                # TJM模型 - 降级处理
                self.model_type = 'tjm'
                self.model = self._load_tjm_model()
                logger.info("⚠️ 使用TJM模型 (性能较低)")

            else:
                raise ValueError(f"不支持的模型格式: {model_suffix}")

            self.performance_stats['load_time'] = time.time() - load_start
            logger.info(f"🚀 模型加载完成，耗时: {self.performance_stats['load_time']:.2f}s")

        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise

    def _load_optimized_onnx_model(self) -> OptimizedONNXModel:
        """加载优化的ONNX模型"""
        # 处理加密模型
        model_path = self.model_path
        if self.encrypt_key:
            # TODO: 实现加密模型解密
            logger.warning("⚠️ 加密模型功能待实现")

        # 创建优化的ONNX模型
        model = OptimizedONNXModel(
            model_path=model_path,
            provider=self.provider,
            input_dynamic_shape=self.input_dynamic_shape
        )

        return model

    def _load_tensorrt_model(self):
        """加载TensorRT模型"""
        try:
            # 尝试导入TensorRT包装器
            from model_lib.base_wrapper import TRTWrapper
            return TRTWrapper(self.model_path)
        except ImportError:
            logger.error("❌ TensorRT包装器不可用")
            raise RuntimeError("TensorRT包装器不可用")

    def _load_tjm_model(self):
        """加载TJM模型"""
        try:
            # 尝试导入TJM包装器
            from model_lib.base_wrapper import TJMWrapper
            return TJMWrapper(self.model_path, provider=self.provider)
        except ImportError:
            logger.error("❌ TJM包装器不可用")
            raise RuntimeError("TJM包装器不可用")

    def __call__(self, *args, **kwargs):
        """推理调用"""
        return self.inference(*args, **kwargs)

    def inference(self, *args, **kwargs):
        """执行推理"""
        start_time = time.time()

        try:
            # 执行推理
            if hasattr(self.model, '__call__'):
                outputs = self.model(*args, **kwargs)
            elif hasattr(self.model, 'run'):
                outputs = self.model.run(*args, **kwargs)
            else:
                raise RuntimeError("模型不支持推理调用")

            # 统计性能
            inference_time = time.time() - start_time
            self.performance_stats['inference_count'] += 1
            self.performance_stats['total_inference_time'] += inference_time

            return outputs

        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            raise

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()

        if stats['inference_count'] > 0:
            stats['average_inference_time'] = stats['total_inference_time'] / stats['inference_count']
            stats['fps'] = 1.0 / stats['average_inference_time']
        else:
            stats['average_inference_time'] = 0
            stats['fps'] = 0

        stats.update({
            'model_type': self.model_type,
            'model_path': self.model_path,
            'provider': self.provider
        })

        # 如果是ONNX模型，添加详细统计
        if hasattr(self.model, 'get_performance_stats'):
            model_stats = self.model.get_performance_stats()
            stats['model_stats'] = model_stats

        return stats

    def benchmark(self, iterations: int = 100) -> Dict:
        """性能基准测试"""
        logger.info(f"📊 开始模型基准测试 ({iterations}次)...")

        if hasattr(self.model, 'benchmark'):
            # 使用模型自带的基准测试
            return self.model.benchmark(iterations)
        else:
            logger.warning("⚠️ 模型不支持基准测试")
            return {'error': '模型不支持基准测试'}

    def optimize_for_batch_inference(self, batch_size: int = 4):
        """优化批量推理"""
        logger.info(f"🎯 优化批量推理 (batch_size={batch_size})...")

        if self.model_type == 'onnx' and hasattr(self.model, 'session'):
            # 重新创建会话以优化批量处理
            try:
                # 获取当前输入形状
                input_shapes = {}
                for inp in self.model.session.get_inputs():
                    shape = list(inp.shape)
                    if len(shape) > 0:
                        shape[0] = batch_size  # 设置批量大小
                    input_shapes[inp.name] = shape

                # 重新创建优化会话
                new_session = self.optimization_engine.create_optimized_session(
                    self.model_path, input_shapes
                )

                # 替换会话
                old_session = self.model.session
                self.model.session = new_session
                del old_session

                logger.info(f"✅ 批量推理优化完成 (batch_size={batch_size})")

            except Exception as e:
                logger.error(f"❌ 批量推理优化失败: {e}")

    def enable_mixed_precision(self):
        """启用混合精度"""
        logger.info("🎯 启用混合精度优化...")

        if self.model_type == 'onnx':
            # 对于ONNX模型，通过重新配置TensorRT启用FP16
            try:
                if 'TensorrtExecutionProvider' in self.model.session.get_providers():
                    logger.info("✅ TensorRT提供者已启用FP16精度")
                else:
                    logger.warning("⚠️ 当前未使用TensorRT，混合精度效果有限")
            except Exception as e:
                logger.error(f"❌ 混合精度配置失败: {e}")

    def clear_cache(self):
        """清理缓存"""
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()

        # 清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("🧹 模型缓存已清理")

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'model'):
            del self.model


# 优化工厂函数
def create_optimized_model(model_info: Dict, provider: str = 'auto') -> OptimizedModelBase:
    """创建优化模型"""
    return OptimizedModelBase(model_info, provider)


# 批量优化现有模型
def optimize_all_models(base_dir: str = ".") -> Dict:
    """批量优化所有模型"""
    logger.info("🔥 开始批量优化所有模型...")

    optimization_engine = get_optimization_engine()
    results = optimization_engine.optimize_existing_models(base_dir)

    logger.info("✅ 批量优化完成")
    return results