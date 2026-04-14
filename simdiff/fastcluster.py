import numpy as np
import torch
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from collections import defaultdict
import time

class FastSpectralClustering:
    def __init__(self, n_clusters=2, sigma=1.0, method='normalized', 
                 use_sparse=True, use_fast_kmeans=True, use_randomized_svd=False,
                 approx_neighbors=50, batch_size=1000, max_iter=100, 
                 normalize_weights=True, weight_threshold=1e-6):
        """
        高效带权重谱聚类算法实现
        
        Args:
            n_clusters: 聚类数量
            sigma: 高斯核的带宽参数
            method: 拉普拉斯矩阵类型 ('unnormalized', 'normalized', 'random_walk')
            use_sparse: 是否使用稀疏矩阵优化
            use_fast_kmeans: 是否使用MiniBatchKMeans
            use_randomized_svd: 是否使用随机化SVD
            approx_neighbors: 近似最近邻数量
            batch_size: MiniBatchKMeans的批大小
            max_iter: 最大迭代次数
            normalize_weights: 是否归一化权重
            weight_threshold: 权重阈值，小于此值的边将被忽略
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.method = method
        self.use_sparse = use_sparse
        self.use_fast_kmeans = use_fast_kmeans
        self.use_randomized_svd = use_randomized_svd
        self.approx_neighbors = approx_neighbors
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.normalize_weights = normalize_weights
        self.weight_threshold = weight_threshold
        
    def build_sparse_graph_from_weighted_edges(self, edges, normalize_weights=None):
        """
        从带权重的边列表构建稀疏图邻接矩阵
        
        Args:
            edges: n×3的数组，每行为[node1, node2, weight]
            normalize_weights: 是否归一化权重
            
        Returns:
            adjacency: 稀疏邻接矩阵
            nodes: 节点列表
        """
        if isinstance(edges, torch.Tensor):
            edges = edges.numpy()
        
        # 确保输入是正确的格式
        if edges.shape[1] != 3:
            raise ValueError(f"期望输入为n×3的边列表（包含权重），但得到形状为{edges.shape}")
        
        # 提取节点和权重
        edge_list = edges[:, :2].astype(int)  # 前两列是节点
        weights = edges[:, 2].astype(float)   # 第三列是权重
        
        # 过滤掉权重过小的边
        valid_edges = weights >= self.weight_threshold
        edge_list = edge_list[valid_edges]
        weights = weights[valid_edges]
        
        if len(edge_list) == 0:
            raise ValueError("所有边的权重都小于阈值，无法构建图")
        
        # 权重归一化
        if normalize_weights is None:
            normalize_weights = self.normalize_weights
            
        if normalize_weights:
            # 将权重归一化到[0, 1]范围
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        # 获取所有唯一节点
        nodes = np.unique(edge_list.flatten())
        n_nodes = len(nodes)
        
        # 创建节点到索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # 构建稀疏矩阵的行、列、数据
        row_indices = []
        col_indices = []
        data = []
        
        for i, (u, v) in enumerate(edge_list):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = weights[i]
            
            # 无向图，添加两个方向的边
            row_indices.extend([u_idx, v_idx])
            col_indices.extend([v_idx, u_idx])
            data.extend([weight, weight])
        
        # 创建稀疏邻接矩阵
        adjacency = csr_matrix((data, (row_indices, col_indices)), 
                              shape=(n_nodes, n_nodes))
        
        return adjacency, nodes
    
    def build_knn_graph(self, adjacency, k=None):
        """
        构建基于权重的k近邻图以减少边的数量
        保留每个节点权重最大的k条边
        """
        if k is None:
            k = min(self.approx_neighbors, adjacency.shape[0] - 1)
        
        n_nodes = adjacency.shape[0]
        
        # 为每个节点只保留k个权重最大的连接
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(n_nodes):
            # 获取第i行的所有非零元素
            start_idx = adjacency.indptr[i]
            end_idx = adjacency.indptr[i + 1]
            
            if end_idx - start_idx <= k:
                # 如果连接数少于k，保留所有连接
                row_indices.extend([i] * (end_idx - start_idx))
                col_indices.extend(adjacency.indices[start_idx:end_idx])
                data.extend(adjacency.data[start_idx:end_idx])
            else:
                # 选择权重最大的k个连接
                weights = adjacency.data[start_idx:end_idx]
                indices = adjacency.indices[start_idx:end_idx]
                
                # 获取top-k索引
                top_k_idx = np.argpartition(weights, -k)[-k:]
                
                row_indices.extend([i] * k)
                col_indices.extend(indices[top_k_idx])
                data.extend(weights[top_k_idx])
        
        # 构建新的稀疏矩阵
        knn_graph = csr_matrix((data, (row_indices, col_indices)), 
                              shape=(n_nodes, n_nodes))
        
        # 确保对称性（取两个方向的最大权重）
        knn_graph = knn_graph.maximum(knn_graph.T)
        
        return knn_graph
    
    def compute_sparse_laplacian(self, adjacency):
        """
        高效计算带权重的稀疏拉普拉斯矩阵
        """
        n_nodes = adjacency.shape[0]
        
        # 计算加权度向量
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        
        if self.method == 'unnormalized':
            # L = D - W
            degree_matrix = diags(degrees, format='csr')
            laplacian = degree_matrix - adjacency
            
        elif self.method == 'normalized':
            # L = I - D^(-1/2) * W * D^(-1/2)
            degrees_sqrt_inv = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)
            degree_sqrt_inv_matrix = diags(degrees_sqrt_inv, format='csr')
            
            normalized_adjacency = degree_sqrt_inv_matrix @ adjacency @ degree_sqrt_inv_matrix
            identity = diags(np.ones(n_nodes), format='csr')
            laplacian = identity - normalized_adjacency
            
        elif self.method == 'random_walk':
            # L = I - D^(-1) * W
            degrees_inv = np.where(degrees > 0, 1.0 / degrees, 0)
            degree_inv_matrix = diags(degrees_inv, format='csr')
            
            normalized_adjacency = degree_inv_matrix @ adjacency
            identity = diags(np.ones(n_nodes), format='csr')
            laplacian = identity - normalized_adjacency
        
        return laplacian
    
    def fast_eigendecomposition(self, laplacian):
        """
        快速特征值分解
        """
        if self.use_randomized_svd and laplacian.shape[0] > 1000:
            # 对于大型矩阵使用随机化SVD
            try:
                # 转换为密集矩阵进行SVD (仅适用于相对较小的矩阵)
                if laplacian.shape[0] < 5000:
                    laplacian_dense = laplacian.toarray()
                    U, s, Vt = randomized_svd(laplacian_dense, 
                                            n_components=self.n_clusters,
                                            random_state=42)
                    # SVD返回的是按降序排列的，我们需要最小的特征值
                    eigenvectors = U[:, -self.n_clusters:]
                else:
                    # 对于非常大的矩阵，仍使用稀疏求解器
                    eigenvalues, eigenvectors = eigsh(laplacian, k=self.n_clusters, 
                                                    which='SM', maxiter=self.max_iter)
            except:
                # 回退到标准方法
                eigenvalues, eigenvectors = eigsh(laplacian, k=self.n_clusters, 
                                                which='SM', maxiter=self.max_iter)
        else:
            # 使用稀疏特征值求解器
            try:
                eigenvalues, eigenvectors = eigsh(laplacian, k=self.n_clusters, 
                                                which='SM', maxiter=self.max_iter)
            except:
                # 如果稀疏求解器失败，转换为密集矩阵
                laplacian_dense = laplacian.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
                idx = np.argsort(eigenvalues)[:self.n_clusters]
                eigenvectors = eigenvectors[:, idx]
        
        return eigenvectors
    
    def fast_kmeans(self, embeddings):
        """
        快速K-means聚类
        """
        if self.use_fast_kmeans and embeddings.shape[0] > 1000:
            # 使用MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters, 
                batch_size=min(self.batch_size, embeddings.shape[0] // 10),
                random_state=42,
                max_iter=self.max_iter,
                n_init=3  # 减少初始化次数
            )
        else:
            # 使用标准KMeans但减少迭代次数
            kmeans = KMeans(
                n_clusters=self.n_clusters, 
                random_state=42,
                max_iter=min(self.max_iter, 300),
                n_init=10
            )
        
        labels = kmeans.fit_predict(embeddings)
        return labels
    
    def build_graph_from_weighted_edges(self, edges):
        """
        从带权重的边列表构建图的邻接矩阵（密集版本）
        
        Args:
            edges: n×3的数组，每行为[node1, node2, weight]
        
        Returns:
            adjacency_matrix: 邻接矩阵
            node_list: 节点列表
        """
        if isinstance(edges, torch.Tensor):
            edges = edges.numpy()
        
        # 确保输入是正确的格式
        if edges.shape[1] != 3:
            raise ValueError(f"期望输入为n×3的边列表（包含权重），但得到形状为{edges.shape}")
        
        # 提取节点和权重
        edge_list = edges[:, :2].astype(int)
        weights = edges[:, 2].astype(float)
        
        # 过滤掉权重过小的边
        valid_edges = weights >= self.weight_threshold
        edge_list = edge_list[valid_edges]
        weights = weights[valid_edges]
        
        if len(edge_list) == 0:
            raise ValueError("所有边的权重都小于阈值，无法构建图")
        
        # 权重归一化
        if self.normalize_weights:
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        # 获取所有唯一节点
        nodes = np.unique(edge_list.flatten())
        n_nodes = len(nodes)
        
        # 创建节点到索引的映射
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # 构建邻接矩阵
        adjacency = np.zeros((n_nodes, n_nodes))
        
        for i, (u, v) in enumerate(edge_list):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            weight = weights[i]
            
            adjacency[u_idx, v_idx] = weight
            adjacency[v_idx, u_idx] = weight  # 无向图
        
        return adjacency, nodes
    
    def compute_laplacian(self, adjacency):
        """
        计算带权重的拉普拉斯矩阵（密集版本）
        
        Args:
            adjacency: 带权重的邻接矩阵
        
        Returns:
            laplacian: 拉普拉斯矩阵
        """
        # 计算加权度矩阵
        degree = np.diag(np.sum(adjacency, axis=1))
        
        if self.method == 'unnormalized':
            # 未归一化拉普拉斯矩阵: L = D - W
            laplacian = degree - adjacency
        elif self.method == 'normalized':
            # 归一化拉普拉斯矩阵: L = D^(-1/2) * (D - W) * D^(-1/2)
            degree_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(degree) + 1e-10))
            laplacian = degree_sqrt_inv @ (degree - adjacency) @ degree_sqrt_inv
        elif self.method == 'random_walk':
            # 随机游走归一化: L = D^(-1) * (D - W)
            degree_inv = np.diag(1.0 / (np.diag(degree) + 1e-10))
            laplacian = degree_inv @ (degree - adjacency)
        
        return laplacian
    
    def fit_predict(self, edges, verbose=False):
        """
        执行带权重的快速谱聚类
        
        Args:
            edges: n×3的边列表，每行为[node1, node2, weight]
            verbose: 是否显示详细时间信息
        
        Returns:
            clusters: list[list] - 每个聚类包含的节点列表
            embeddings: 谱嵌入矩阵
            nodes: 节点列表
        """
        if verbose:
            start_time = time.time()
        
        # 验证输入格式
        if isinstance(edges, (list, tuple)):
            edges = np.array(edges)
        elif isinstance(edges, torch.Tensor):
            edges = edges.numpy()
        
        if edges.shape[1] != 3:
            raise ValueError(f"边列表必须是n×3的格式[node1, node2, weight]，但得到形状{edges.shape}")
        
        # 1. 构建带权重的稀疏图
        if verbose:
            print("步骤1: 构建带权重的图...")
            step_start = time.time()
        
        if self.use_sparse:
            adjacency, nodes = self.build_sparse_graph_from_weighted_edges(edges)
        else:
            adjacency, nodes = self.build_graph_from_weighted_edges(edges)
            adjacency = csr_matrix(adjacency)  # 转换为稀疏矩阵
        
        if verbose:
            print(f"  图构建完成，节点数: {len(nodes)}, 边数: {len(edges)}")
            print(f"  权重范围: [{adjacency.data.min():.4f}, {adjacency.data.max():.4f}]")
            print(f"  用时: {time.time() - step_start:.3f}秒")
        
        # 2. 可选：基于权重构建k近邻图
        if adjacency.shape[0] > 1000 and self.approx_neighbors < adjacency.shape[0]:
            if verbose:
                print("步骤2: 构建基于权重的k近邻图...")
                step_start = time.time()
            
            adjacency = self.build_knn_graph(adjacency, self.approx_neighbors)
            
            if verbose:
                print(f"  k近邻图构建完成，k={self.approx_neighbors}")
                print(f"  用时: {time.time() - step_start:.3f}秒")
        
        # 3. 计算带权重的拉普拉斯矩阵
        if verbose:
            print("步骤3: 计算带权重的拉普拉斯矩阵...")
            step_start = time.time()
        
        laplacian = self.compute_sparse_laplacian(adjacency)
        
        if verbose:
            print(f"  拉普拉斯矩阵计算完成，方法: {self.method}")
            print(f"  用时: {time.time() - step_start:.3f}秒")
        
        # 4. 快速特征值分解
        if verbose:
            print("步骤4: 特征值分解...")
            step_start = time.time()
        
        eigenvectors = self.fast_eigendecomposition(laplacian)
        
        if verbose:
            print(f"  特征值分解完成，使用随机化SVD: {self.use_randomized_svd}")
            print(f"  用时: {time.time() - step_start:.3f}秒")
        
        # 5. 构建嵌入矩阵
        embeddings = eigenvectors
        
        # 6. 快速K-means聚类
        if verbose:
            print("步骤5: K-means聚类...")
            step_start = time.time()
        
        labels = self.fast_kmeans(embeddings)
        
        if verbose:
            print(f"  K-means聚类完成，使用MiniBatch: {self.use_fast_kmeans}")
            print(f"  用时: {time.time() - step_start:.3f}秒")
        
        # 7. 将结果转换为list[list]格式
        clusters = self._labels_to_clusters(labels, nodes)
        
        if verbose:
            total_time = time.time() - start_time
            print(f"总用时: {total_time:.3f}秒")
        
        return clusters, embeddings, nodes
    
    def _labels_to_clusters(self, labels, nodes):
        """
        将聚类标签转换为list[list]格式
        
        Args:
            labels: 聚类标签数组
            nodes: 节点列表
        
        Returns:
            clusters: list[list] - 每个聚类包含的节点列表
        """
        clusters = [[] for _ in range(self.n_clusters)]
        
        for i, label in enumerate(labels):
            clusters[label].append(nodes[i])
        
        # 移除空的聚类
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        
        return clusters

    # 为了向后兼容，保留原有方法但添加权重支持
    def build_graph_from_edges(self, edges, weighted=True, weights=None):
        """
        兼容原有接口，支持两种输入格式：
        1. edges为n×2，weights为单独数组
        2. edges为n×3，包含权重
        """
        if isinstance(edges, torch.Tensor):
            edges = edges.numpy()
        
        if edges.shape[1] == 3:
            # 新格式：n×3包含权重
            return self.build_graph_from_weighted_edges(edges)
        elif edges.shape[1] == 2:
            # 旧格式：n×2 + 权重数组
            if weights is not None:
                # 合并为n×3格式
                weighted_edges = np.column_stack([edges, weights])
                return self.build_graph_from_weighted_edges(weighted_edges)
            else:
                # 没有权重，使用单位权重
                weights = np.ones(len(edges))
                weighted_edges = np.column_stack([edges, weights])
                return self.build_graph_from_weighted_edges(weighted_edges)
        else:
            raise ValueError(f"不支持的边格式，形状为{edges.shape}")