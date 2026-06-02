#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于NumPy的张量运算实现
此模块使用NumPy实现各种张量运算，
包括基本算术运算、内积、缩并和einsum。
"""


import numpy as np
from typing import Union, Tuple, Optional


class TensorCalculator:
    """
    用于执行各种张量运算的类（使用NumPy）
    """
    
    @staticmethod
    def add(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        对两个张量进行加法运算
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            tensor1 + tensor2 的结果
        """
        return np.add(tensor1, tensor2)
    
    @staticmethod
    def subtract(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        对两个张量进行减法运算
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            tensor1 - tensor2 的结果
        """
        return np.subtract(tensor1, tensor2)
    
    @staticmethod
    def multiply(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        对两个张量进行逐元素乘法运算
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            tensor1 * tensor2 的结果
        """
        return np.multiply(tensor1, tensor2)
    
    @staticmethod
    def divide(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        对两个张量进行逐元素除法运算
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            tensor1 / tensor2 的结果
        """
        return np.divide(tensor1, tensor2)
    
    @staticmethod
    def inner_product(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        计算两个张量的内积
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            两个张量的内积
        """
        return np.inner(tensor1, tensor2)
    
    @staticmethod
    def dot_product(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        计算两个张量的点积
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            两个张量的点积
        """
        return np.dot(tensor1, tensor2)
    
    @staticmethod
    def matrix_multiply(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
        """
        计算两个张量的矩阵乘法
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            
        返回:
            矩阵乘法的结果
        """
        return np.matmul(tensor1, tensor2)
    
    @staticmethod
    def einsum(equation: str, *operands) -> np.ndarray:
        """
        张量运算的爱因斯坦求和约定
        
        参数:
            equation: 爱因斯坦求和方程字符串
            operands: 要操作的张量
            
        返回:
            爱因斯坦求和运算的结果
        """
        return np.einsum(equation, *operands)
    
    @staticmethod
    def trace(tensor: np.ndarray, axis1: int = 0, axis2: int = 1) -> np.ndarray:
        """
        计算沿指定轴的张量迹
        
        参数:
            tensor: 输入张量
            axis1: 用于迹计算的第一个轴
            axis2: 用于迹计算的第二个轴
            
        返回:
            沿指定轴的张量迹
        """
        return np.trace(tensor, axis1=axis1, axis2=axis2)
    
    @staticmethod
    def transpose(tensor: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        转置张量
        
        参数:
            tensor: 输入张量
            axes: 轴的排列
            
        返回:
            转置后的张量
        """
        return np.transpose(tensor, axes=axes)
    
    @staticmethod
    def contract(tensor1: np.ndarray, tensor2: np.ndarray, 
                 axes1: Union[int, Tuple[int, ...]], 
                 axes2: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        沿指定轴缩并两个张量
        
        参数:
            tensor1: 第一个张量
            tensor2: 第二个张量
            axes1: tensor1要缩并的轴
            axes2: tensor2要缩并的轴
            
        返回:
            缩并后的张量
        """
        return np.tensordot(tensor1, tensor2, axes=(axes1, axes2))


def tensor_add(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """
    两个张量相加的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        
    返回:
        tensor1 + tensor2 的结果
    """
    calculator = TensorCalculator()
    return calculator.add(tensor1, tensor2)


def tensor_subtract(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """
    两个张量相减的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        
    返回:
        tensor1 - tensor2 的结果
    """
    calculator = TensorCalculator()
    return calculator.subtract(tensor1, tensor2)


def tensor_multiply(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """
    两个张量逐元素相乘的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        
    返回:
        tensor1 * tensor2 的结果
    """
    calculator = TensorCalculator()
    return calculator.multiply(tensor1, tensor2)


def tensor_divide(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """
    两个张量逐元素相除的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        
    返回:
        tensor1 / tensor2 的结果
    """
    calculator = TensorCalculator()
    return calculator.divide(tensor1, tensor2)


def tensor_inner_product(tensor1: np.ndarray, tensor2: np.ndarray) -> np.ndarray:
    """
    计算两个张量内积的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        
    返回:
        两个张量的内积
    """
    calculator = TensorCalculator()
    return calculator.inner_product(tensor1, tensor2)


def tensor_einsum(equation: str, *operands) -> np.ndarray:
    """
    爱因斯坦求和的便捷函数
    
    参数:
        equation: 爱因斯坦求和方程字符串
        operands: 要操作的张量
        
    返回:
        爱因斯坦求和运算的结果
    """
    calculator = TensorCalculator()
    return calculator.einsum(equation, *operands)


def tensor_contract(tensor1: np.ndarray, tensor2: np.ndarray, 
                   axes1: Union[int, Tuple[int, ...]], 
                   axes2: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    缩并两个张量的便捷函数
    
    参数:
        tensor1: 第一个张量
        tensor2: 第二个张量
        axes1: tensor1要缩并的轴
        axes2: tensor2要缩并的轴
        
    返回:
        缩并后的张量
    """
    calculator = TensorCalculator()
    return calculator.contract(tensor1, tensor2, axes1, axes2)


# 示例用法和测试
if __name__ == "__main__":
    # 创建示例张量
    t1 = np.array([[1, 2], [3, 4]])
    t2 = np.array([[5, 6], [7, 8]])
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print("示例张量运算:")
    print(f"张量 1:\n{t1}")
    print(f"张量 2:\n{t2}")
    
    print(f"\n加法:\n{tensor_add(t1, t2)}")
    print(f"减法:\n{tensor_subtract(t1, t2)}")
    print(f"逐元素乘法:\n{tensor_multiply(t1, t2)}")
    print(f"逐元素除法:\n{tensor_divide(t1, t2)}")
    
    print(f"\n向量的内积:\n{tensor_inner_product(v1, v2)}")
    print(f"矩阵的点积:\n{TensorCalculator.dot_product(t1, t2)}")
    print(f"矩阵乘法:\n{TensorCalculator.matrix_multiply(t1, t2)}")
    
    # 爱因斯坦求和示例
    print(f"\n爱因斯坦求和 'ij->ji' (转置):\n{tensor_einsum('ij->ji', t1)}")
    print(f"爱因斯坦求和 'ii->i' (对角线):\n{tensor_einsum('ii->i', t1)}")
    print(f"爱因斯坦求和 'ij,jk->ik' (矩阵乘法):\n{tensor_einsum('ij,jk->ik', t1, t2)}")
    
    # 缩并示例
    print(f"\n张量缩并 (t1的最后一维与t2的第一维):\n{tensor_contract(t1, t2, 1, 0)}")
    
    # 迹
    print(f"\nt1的迹:\n{TensorCalculator.trace(t1)}")
    
    # 转置
    print(f"\nt1的转置:\n{TensorCalculator.transpose(t1)}")