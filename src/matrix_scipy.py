# -*- coding: utf-8 -*-
"""
基于SciPy的稀疏矩阵和线性方程组求解实现
此模块使用SciPy实现稀疏矩阵的构造、操作和线性方程组的求解，
以及通过Matplotlib进行可视化。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
from typing import Tuple, Union, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')


class SparseMatrixCalculator:
    """
    用于执行稀疏矩阵运算的类（使用SciPy）
    """
    
    @staticmethod
    def create_csr_matrix(data: np.ndarray, indices: np.ndarray, 
                         indptr: np.ndarray, shape: Tuple[int, int]) -> sparse.csr_matrix:
        """
        使用压缩行格式(CSR)创建稀疏矩阵
        
        参数:
            data: 非零元素的值
            indices: 非零元素的列索引
            indptr: 行指针数组
            shape: 矩阵的形状
            
        返回:
            CSR格式的稀疏矩阵
        """
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    
    @staticmethod
    def create_coo_matrix(row: np.ndarray, col: np.ndarray, 
                         data: np.ndarray, shape: Tuple[int, int]) -> sparse.coo_matrix:
        """
        使用坐标格式(COO)创建稀疏矩阵
        
        参数:
            row: 非零元素的行索引
            col: 非零元素的列索引
            data: 非零元素的值
            shape: 矩阵的形状
            
        返回:
            COO格式的稀疏矩阵
        """
        return sparse.coo_matrix((data, (row, col)), shape=shape)
    
    @staticmethod
    def create_lil_matrix(shape: Tuple[int, int]) -> sparse.lil_matrix:
        """
        创建列表格式(LIL)的稀疏矩阵，便于增量构造
        
        参数:
            shape: 矩阵的形状
            
        返回:
            LIL格式的稀疏矩阵
        """
        return sparse.lil_matrix(shape)
    
    @staticmethod
    def from_dense(array: np.ndarray) -> sparse.csr_matrix:
        """
        从密集数组转换为稀疏矩阵
        
        参数:
            array: 密集的NumPy数组
            
        返回:
            CSR格式的稀疏矩阵
        """
        return sparse.csr_matrix(array)
    
    @staticmethod
    def to_dense(sparse_mat: sparse.spmatrix) -> np.ndarray:
        """
        将稀疏矩阵转换为密集数组
        
        参数:
            sparse_mat: 稀疏矩阵
            
        返回:
            密集的NumPy数组
        """
        return sparse_mat.toarray()
    
    @staticmethod
    def get_sparsity(sparse_mat: sparse.spmatrix) -> float:
        """
        计算稀疏矩阵的稀疏度（零元素的比例）
        
        参数:
            sparse_mat: 稀疏矩阵
            
        返回:
            稀疏度（0到1之间）
        """
        total_elements = sparse_mat.shape[0] * sparse_mat.shape[1]
        non_zero_elements = sparse_mat.nnz
        return 1.0 - (non_zero_elements / total_elements)
    
    @staticmethod
    def sparse_multiply(mat1: sparse.spmatrix, mat2: sparse.spmatrix) -> sparse.csr_matrix:
        """
        两个稀疏矩阵的乘法
        
        参数:
            mat1: 第一个稀疏矩阵
            mat2: 第二个稀疏矩阵
            
        返回:
            乘积矩阵（CSR格式）
        """
        result = mat1 * mat2
        return result.tocsr()
    
    @staticmethod
    def sparse_add(mat1: sparse.spmatrix, mat2: sparse.spmatrix) -> sparse.csr_matrix:
        """
        两个稀疏矩阵的加法
        
        参数:
            mat1: 第一个稀疏矩阵
            mat2: 第二个稀疏矩阵
            
        返回:
            和矩阵（CSR格式）
        """
        result = mat1 + mat2
        return result.tocsr()
    
    @staticmethod
    def solve_linear_system(A: Union[sparse.spmatrix, np.ndarray], 
                           b: np.ndarray, method: str = 'auto') -> np.ndarray:
        """
        求解线性方程组 Ax = b
        
        参数:
            A: 系数矩阵（可以是稀疏或密集）
            b: 右侧向量
            method: 求解方法 ('auto', 'direct', 'lsqr', 'gmres', 'bicg')
            
        返回:
            解向量x
        """
        if not isinstance(A, sparse.spmatrix):
            A = sparse.csr_matrix(A)
        else:
            A = A.tocsr()
        
        if method == 'auto':
            try:
                x = sp_linalg.spsolve(A, b)
            except Exception:
                x = sp_linalg.lsqr(A, b)[0]
        elif method == 'direct':
            x = sp_linalg.spsolve(A, b)
        elif method == 'lsqr':
            x = sp_linalg.lsqr(A, b)[0]
        elif method == 'gmres':
            x, info = sp_linalg.gmres(A, b)
            if info != 0:
                print(f"警告：GMRES未收敛，返回最佳近似")
        elif method == 'bicg':
            x, info = sp_linalg.bicg(A, b)
            if info != 0:
                print(f"警告：BICG未收敛，返回最佳近似")
        else:
            raise ValueError(f"未知的求解方法: {method}")
        
        return x
    
    @staticmethod
    def solve_least_squares(A: Union[sparse.spmatrix, np.ndarray], 
                           b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用最小二乘法求解超定或欠定方程组
        
        参数:
            A: 系数矩阵（m x n）
            b: 右侧向量（m,）
            
        返回:
            (解向量x, 残差范数)
        """
        if not isinstance(A, sparse.spmatrix):
            A = sparse.csr_matrix(A)
        else:
            A = A.tocsr()
        
        result = sp_linalg.lsqr(A, b)
        x = result[0]
        r1norm = result[3]
        return x, r1norm
    
    @staticmethod
    def matrix_norm(sparse_mat: sparse.spmatrix, norm_type: str = 'fro') -> float:
        """
        计算稀疏矩阵的范数
        
        参数:
            sparse_mat: 稀疏矩阵
            norm_type: 范数类型 ('fro', '1', 'inf', '2')
            
        返回:
            矩阵的范数
        """
        if norm_type == 'fro':
            return np.sqrt((sparse_mat.multiply(sparse_mat)).sum())
        elif norm_type == '1':
            return max(np.array(sparse_mat.sum(axis=0)).flatten())
        elif norm_type == 'inf':
            return max(np.array(sparse_mat.sum(axis=1)).flatten())
        elif norm_type == '2':
            return sp_linalg.norm(sparse_mat, 2)
        else:
            raise ValueError(f"未知的范数类型: {norm_type}")
    
    @staticmethod
    def get_matrix_stats(sparse_mat: sparse.spmatrix) -> Dict[str, Union[int, float]]:
        """
        获取稀疏矩阵的统计信息
        
        参数:
            sparse_mat: 稀疏矩阵
            
        返回:
            包含统计信息的字典
        """
        stats = {
            'shape': sparse_mat.shape,
            'nnz': sparse_mat.nnz,
            'size': sparse_mat.shape[0] * sparse_mat.shape[1],
            'sparsity': 1.0 - (sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1])),
            'memory_usage_bytes': sparse_mat.data.nbytes + sparse_mat.indices.nbytes + sparse_mat.indptr.nbytes
        }
        return stats


class MatrixVisualizer:
    """
    用于矩阵可视化的类
    """
    
    @staticmethod
    def plot_sparse_matrix(sparse_mat: sparse.spmatrix, title: str = "Sparse Matrix Pattern", 
                          figsize: Tuple[int, int] = (10, 8), cmap: str = 'Blues',
                          show: bool = True) -> None:
        """
        Plot the sparsity pattern of a sparse matrix
        
        Parameters:
            sparse_mat: sparse matrix
            title: plot title
            figsize: figure size
            cmap: colormap
            show: whether to display immediately (True) or defer (False)
        """
        fig, ax = plt.subplots(figsize=figsize)
        coo = sparse_mat.tocoo()
        
        scatter = ax.scatter(coo.col, coo.row, c=coo.data, cmap=cmap, s=50, marker='s')
        
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Non-zero Value')
        
        plt.tight_layout()
        if show:
            plt.show()
    
    @staticmethod
    def plot_matrix_heatmap(array: np.ndarray, title: str = "Matrix Heatmap", 
                           figsize: Tuple[int, int] = (10, 8),
                           show: bool = True) -> None:
        """
        Plot a heatmap of a matrix
        
        Parameters:
            array: matrix array
            title: plot title
            figsize: figure size
            show: whether to display immediately (True) or defer (False)
        """
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(array, cmap='viridis', aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Element Value')
        
        plt.tight_layout()
        if show:
            plt.show()
    
    @staticmethod
    def plot_sparsity_comparison(matrices: List[Tuple[sparse.spmatrix, str]], 
                                figsize: Tuple[int, int] = (12, 6),
                                show: bool = True) -> None:
        """
        Compare sparsity of multiple matrices
        
        Parameters:
            matrices: list of (matrix, label) tuples
            figsize: figure size
            show: whether to display immediately (True) or defer (False)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        labels = [label for _, label in matrices]
        nnz_values = [mat.nnz for mat, _ in matrices]
        total_values = [mat.shape[0] * mat.shape[1] for mat, _ in matrices]
        sparsities = [1.0 - (nnz / total) for nnz, total in zip(nnz_values, total_values)]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(matrices)))
        
        ax1.bar(labels, nnz_values, color=colors)
        ax1.set_ylabel('Number of Non-zero Elements')
        ax1.set_title('Non-zero Elements Comparison')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(labels, sparsities, color=colors)
        ax2.set_ylabel('Sparsity')
        ax2.set_title('Sparsity Comparison')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if show:
            plt.show()
    
    @staticmethod
    def plot_solution_convergence(solutions_history: List[np.ndarray], 
                                 title: str = "Solution Convergence Process", 
                                 figsize: Tuple[int, int] = (10, 6),
                                 show: bool = True) -> None:
        """
        Plot the convergence of solutions during iterative solving
        
        Parameters:
            solutions_history: history of solution vectors during iterations
            title: plot title
            figsize: figure size
            show: whether to display immediately (True) or defer (False)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, solution in enumerate(solutions_history):
            ax.plot(solution, marker='o', label=f'Iteration {i}', alpha=0.7)
        
        ax.set_xlabel('Unknown Index')
        ax.set_ylabel('Solution Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if show:
            plt.show()
    
    @staticmethod
    def show_all() -> None:
        """Display all created figures simultaneously"""
        plt.show()


def create_tridiagonal_matrix(n: int, a: float, b: float, c: float) -> sparse.csr_matrix:
    """
    创建三对角稀疏矩阵
    
    参数:
        n: 矩阵大小
        a: 下对角线元素
        b: 主对角线元素
        c: 上对角线元素
        
    返回:
        三对角稀疏矩阵（CSR格式）
    """
    diagonals = [a * np.ones(n - 1), b * np.ones(n), c * np.ones(n - 1)]
    offsets = [-1, 0, 1]
    return sparse.diags(diagonals, offsets, shape=(n, n), format='csr')


def create_laplacian_matrix(grid_size: Tuple[int, int]) -> sparse.csr_matrix:
    """
    创建二维网格的拉普拉斯矩阵
    
    参数:
        grid_size: 网格大小 (m, n)
        
    返回:
        拉普拉斯稀疏矩阵（CSR格式）
    """
    m, n = grid_size
    total_nodes = m * n
    
    lil = sparse.lil_matrix((total_nodes, total_nodes))
    
    for i in range(m):
        for j in range(n):
            node = i * n + j
            lil[node, node] = 4
            
            if i > 0:
                lil[node, (i - 1) * n + j] = -1
            if i < m - 1:
                lil[node, (i + 1) * n + j] = -1
            if j > 0:
                lil[node, i * n + (j - 1)] = -1
            if j < n - 1:
                lil[node, i * n + (j + 1)] = -1
    
    return lil.tocsr()


# Example usage and tests
if __name__ == "__main__":
    print("=" * 60)
    print("SciPy Sparse Matrix and Linear Solver Demo")
    print("=" * 60)
    
    # Example 1: Create and operate on sparse matrices
    print("\n[Example 1] Creating and Operating on Sparse Matrices")
    print("-" * 60)
    
    n = 100
    mat_tri = create_tridiagonal_matrix(n, -1, 4, -1)
    print(f"Tridiagonal matrix size: {mat_tri.shape}")
    print(f"Number of non-zero elements: {mat_tri.nnz}")
    stats = SparseMatrixCalculator.get_matrix_stats(mat_tri)
    print(f"Sparsity: {stats['sparsity']:.4f}")
    print(f"Memory usage: {stats['memory_usage_bytes']} bytes")
    
    # Example 2: Solve linear systems
    print("\n[Example 2] Solving Linear System Ax = b")
    print("-" * 60)
    
    A = create_tridiagonal_matrix(50, -1, 2, -1)
    b = np.ones(50)
    
    x = SparseMatrixCalculator.solve_linear_system(A, b)
    residual = np.linalg.norm(A.toarray() @ x - b)
    print(f"System size: {A.shape}")
    print(f"Solution first 5 components: {x[:5]}")
    print(f"Residual norm: {residual:.2e}")
    
    # Example 3: Least squares problem
    print("\n[Example 3] Least Squares Problem")
    print("-" * 60)
    
    A_overdetermined = sparse.csr_matrix(np.random.randn(10, 5))
    b_overdetermined = np.random.randn(10)
    
    x_ls, residual_ls = SparseMatrixCalculator.solve_least_squares(A_overdetermined, b_overdetermined)
    print(f"Overdetermined system size: {A_overdetermined.shape}")
    print(f"Least squares solution: {x_ls}")
    print(f"Residual: {residual_ls:.4f}")
    
    # Example 4: Laplacian matrix
    print("\n[Example 4] 2D Grid Laplacian Matrix")
    print("-" * 60)
    
    grid_size = (10, 10)
    lap_mat = create_laplacian_matrix(grid_size)
    print(f"Laplacian matrix size: {lap_mat.shape}")
    print(f"Number of non-zero elements: {lap_mat.nnz}")
    print(f"Sparsity: {SparseMatrixCalculator.get_sparsity(lap_mat):.4f}")
    
    # Example 5: Matrix visualization - Display all plots simultaneously
    print("\n[Example 5] Matrix Visualization")
    print("-" * 60)
    print("Creating all visualization plots...")
    
    visualizer = MatrixVisualizer()
    
    # Create plot 1: Tridiagonal sparse matrix pattern
    print("  1. Sparse matrix pattern...")
    visualizer.plot_sparse_matrix(
        mat_tri, 
        title="Tridiagonal Sparse Matrix Pattern",
        show=False
    )
    
    # Create plot 2: Linear solution heatmap
    print("  2. Linear solution heatmap...")
    visualizer.plot_matrix_heatmap(
        x.reshape(1, -1), 
        title="Linear System Solution",
        show=False
    )
    
    # Create plot 3: Sparsity comparison
    print("  3. Sparsity comparison...")
    visualizer.plot_sparsity_comparison(
        [(mat_tri, "Tridiagonal"), (lap_mat, "Laplacian")],
        figsize=(12, 5),
        show=False
    )
    
    # Create plot 4: Solution convergence
    print("  4. Solution convergence...")
    solutions_history = [x, x * 0.95, x * 1.05]
    visualizer.plot_solution_convergence(
        solutions_history,
        title="Linear System Solution Convergence",
        show=False
    )
    
    print("\nDisplaying all plots simultaneously...")
    visualizer.show_all()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
