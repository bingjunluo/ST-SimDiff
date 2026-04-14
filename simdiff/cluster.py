import torch
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set
import matplotlib.pyplot as plt
import networkx as nx

class GraphClustering:
    def __init__(self, edges: torch.Tensor):
        """
        初始化图聚类器
        Args:
            edges: n*3的tensor，前两列为节点编号，第三列为边权重
        """
        self.edges = edges
        self.nodes = torch.unique(edges[:, :2]).int()
        self.num_nodes = len(self.nodes)
        self.num_edges = edges.shape[0]
        
        # 构建邻接表
        self.adj_list = defaultdict(list)
        self.edge_weights = {}
        
        for i in range(self.num_edges):
            u, v, w = edges[i, 0].item(), edges[i, 1].item(), edges[i, 2].item()
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
            self.edge_weights[(min(u, v), max(u, v))] = w
    
    def threshold_clustering(self, threshold: float) -> List[Set[int]]:
        """
        基于权重阈值的聚类：只保留权重大于阈值的边，然后找连通分量
        Args:
            threshold: 权重阈值
        Returns:
            聚类结果，每个聚类是一个节点集合
        """
        # 构建过滤后的邻接表
        filtered_adj = defaultdict(list)
        
        for i in range(self.num_edges):
            u, v, w = self.edges[i, 0].item(), self.edges[i, 1].item(), self.edges[i, 2].item()
            if w >= threshold:
                filtered_adj[u].append(v)
                filtered_adj[v].append(u)
        
        # 使用DFS找连通分量
        visited = set()
        clusters = []
        
        for node in self.nodes:
            node = node.item()
            if node not in visited:
                cluster = set()
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        for neighbor in filtered_adj[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def hierarchical_clustering(self, linkage='average') -> List[Tuple[float, List[Set[int]]]]:
        """
        层次聚类：从最高权重的边开始，逐步加入边形成聚类
        Args:
            linkage: 链接方式 ('single', 'average', 'complete')
        Returns:
            每个层级的聚类结果，包含阈值和对应的聚类
        """
        # 按权重排序边
        edge_list = []
        for i in range(self.num_edges):
            u, v, w = self.edges[i, 0].item(), self.edges[i, 1].item(), self.edges[i, 2].item()
            edge_list.append((w, u, v))
        
        edge_list.sort(reverse=True)  # 从高权重到低权重
        
        results = []
        used_thresholds = set()
        
        for weight, _, _ in edge_list:
            if weight not in used_thresholds:
                clusters = self.threshold_clustering(weight)
                results.append((weight, clusters))
                used_thresholds.add(weight)
        
        return results
    
    def modularity_clustering(self, resolution=1.0) -> List[Set[int]]:
        """
        基于模块度的聚类算法（简化版Louvain算法）
        Args:
            resolution: 分辨率参数
        Returns:
            聚类结果
        """
        # 初始化：每个节点为一个聚类
        node_to_cluster = {node.item(): i for i, node in enumerate(self.nodes)}
        clusters = [{node.item()} for node in self.nodes]
        
        # 计算总权重
        total_weight = self.edges[:, 2].sum().item()
        
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for node in self.nodes:
                node = node.item()
                current_cluster = node_to_cluster[node]
                best_cluster = current_cluster
                best_gain = 0
                
                # 尝试移动到邻居的聚类
                neighbor_clusters = set()
                for neighbor in self.adj_list[node]:
                    neighbor_clusters.add(node_to_cluster[neighbor])
                
                for target_cluster in neighbor_clusters:
                    if target_cluster != current_cluster:
                        gain = self._calculate_modularity_gain(
                            node, current_cluster, target_cluster, 
                            node_to_cluster, total_weight, resolution
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_cluster = target_cluster
                
                # 如果找到更好的聚类，则移动
                if best_cluster != current_cluster and best_gain > 0:
                    clusters[current_cluster].remove(node)
                    clusters[best_cluster].add(node)
                    node_to_cluster[node] = best_cluster
                    improved = True
        
        # 移除空聚类
        final_clusters = [cluster for cluster in clusters if cluster]
        return final_clusters
    
    def _calculate_modularity_gain(self, node, from_cluster, to_cluster, 
                                 node_to_cluster, total_weight, resolution):
        """计算移动节点到新聚类的模块度增益"""
        # 简化的模块度增益计算
        gain = 0
        
        # 计算与目标聚类的连接权重
        for neighbor in self.adj_list[node]:
            edge_key = (min(node, neighbor), max(node, neighbor))
            weight = self.edge_weights.get(edge_key, 0)
            
            if node_to_cluster[neighbor] == to_cluster:
                gain += weight
            elif node_to_cluster[neighbor] == from_cluster:
                gain -= weight
        
        return gain / total_weight if total_weight > 0 else 0
    
    def louvain_clustering(self, resolution=1.0, max_iterations=100) -> List[Set[int]]:
        """
        Louvain算法进行社区检测
        Args:
            resolution: 分辨率参数，控制社区大小
            max_iterations: 最大迭代次数
        Returns:
            聚类结果
        """
        # 第一阶段：节点级优化
        node_to_community = {node.item(): node.item() for node in self.nodes}
        communities = {node.item(): {node.item()} for node in self.nodes}
        
        # 预计算节点度和权重
        node_weights = {}
        total_weight = 0
        
        for node in self.nodes:
            node = node.item()
            weight = 0
            for neighbor in self.adj_list[node]:
                edge_key = (min(node, neighbor), max(node, neighbor))
                weight += self.edge_weights.get(edge_key, 0)
            node_weights[node] = weight
            total_weight += weight
        
        total_weight = total_weight / 2  # 因为每条边被计算了两次
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 随机化节点顺序以避免偏置
            nodes_shuffled = torch.randperm(len(self.nodes))
            
            for i in nodes_shuffled:
                node = self.nodes[i].item()
                current_community = node_to_community[node]
                best_community = current_community
                best_modularity_gain = 0
                
                # 获取邻居社区
                neighbor_communities = set()
                for neighbor in self.adj_list[node]:
                    neighbor_communities.add(node_to_community[neighbor])
                
                # 计算移动到每个邻居社区的模块度增益
                for target_community in neighbor_communities:
                    if target_community != current_community:
                        gain = self._louvain_modularity_gain(
                            node, current_community, target_community,
                            node_to_community, node_weights, total_weight, resolution
                        )
                        
                        if gain > best_modularity_gain:
                            best_modularity_gain = gain
                            best_community = target_community
                
                # 如果找到更好的社区，则移动节点
                if best_community != current_community and best_modularity_gain > 0:
                    # 从原社区移除
                    communities[current_community].remove(node)
                    if not communities[current_community]:
                        del communities[current_community]
                    
                    # 添加到新社区
                    if best_community not in communities:
                        communities[best_community] = set()
                    communities[best_community].add(node)
                    
                    node_to_community[node] = best_community
                    improved = True
        
        # 第二阶段：社区级优化（可选，这里简化处理）
        final_communities = list(communities.values())
        return final_communities
    
    def _louvain_modularity_gain(self, node, from_community, to_community,
                                node_to_community, node_weights, total_weight, resolution):
        """计算Louvain算法中的模块度增益"""
        if total_weight == 0:
            return 0
        
        # 计算节点与目标社区和原社区的连接权重
        k_i_in_to = 0  # 节点到目标社区的权重
        k_i_in_from = 0  # 节点到原社区的权重
        
        for neighbor in self.adj_list[node]:
            edge_key = (min(node, neighbor), max(node, neighbor))
            weight = self.edge_weights.get(edge_key, 0)
            
            if node_to_community[neighbor] == to_community:
                k_i_in_to += weight
            elif node_to_community[neighbor] == from_community:
                k_i_in_from += weight
        
        # 计算社区总权重
        sigma_tot_to = sum(node_weights[n] for n in node_to_community 
                          if node_to_community[n] == to_community)
        sigma_tot_from = sum(node_weights[n] for n in node_to_community 
                            if node_to_community[n] == from_community)
        
        k_i = node_weights[node]
        
        # 模块度增益公式
        delta_q = (k_i_in_to - k_i_in_from) / total_weight - \
                  resolution * k_i * (sigma_tot_to - sigma_tot_from + k_i) / (2 * total_weight ** 2)
        
        return delta_q
    
    def label_propagation_clustering(self, max_iterations=100, seed=None) -> List[Set[int]]:
        """
        标签传播算法进行社区检测
        Args:
            max_iterations: 最大迭代次数
            seed: 随机种子
        Returns:
            聚类结果
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 初始化：每个节点的标签为自己
        node_labels = {node.item(): node.item() for node in self.nodes}
        
        converged = False
        iteration = 0
        
        while not converged and iteration < max_iterations:
            converged = True
            iteration += 1
            
            # 随机化节点顺序
            nodes_shuffled = torch.randperm(len(self.nodes))
            
            for i in nodes_shuffled:
                node = self.nodes[i].item()
                
                # 计算邻居标签的权重
                label_weights = defaultdict(float)
                
                for neighbor in self.adj_list[node]:
                    edge_key = (min(node, neighbor), max(node, neighbor))
                    weight = self.edge_weights.get(edge_key, 0)
                    neighbor_label = node_labels[neighbor]
                    label_weights[neighbor_label] += weight
                
                if label_weights:
                    # 选择权重最大的标签
                    max_weight = max(label_weights.values())
                    best_labels = [label for label, weight in label_weights.items() 
                                 if weight == max_weight]
                    
                    # 如果有多个相同权重的标签，随机选择一个
                    if best_labels:  # 确保best_labels不为空
                        new_label = np.random.choice(best_labels)
                        
                        if new_label != node_labels[node]:
                            node_labels[node] = new_label
                            converged = False
                # 如果没有邻居或邻居权重为0，保持原标签不变
        
        # 构建聚类结果
        clusters = defaultdict(set)
        for node, label in node_labels.items():
            clusters[label].add(node)
        
        return list(clusters.values())
    
    def asynchronous_label_propagation(self, max_iterations=100, seed=None) -> List[Set[int]]:
        """
        异步标签传播算法（改进版本）
        在每次迭代中使用最新的标签信息
        Args:
            max_iterations: 最大迭代次数  
            seed: 随机种子
        Returns:
            聚类结果
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 初始化标签
        node_labels = {node.item(): node.item() for node in self.nodes}
        
        for iteration in range(max_iterations):
            old_labels = node_labels.copy()
            
            # 随机化节点顺序
            nodes_shuffled = torch.randperm(len(self.nodes))
            
            for i in nodes_shuffled:
                node = self.nodes[i].item()
                
                # 使用当前最新的标签计算邻居标签权重
                label_weights = defaultdict(float)
                
                for neighbor in self.adj_list[node]:
                    edge_key = (min(node, neighbor), max(node, neighbor))
                    weight = self.edge_weights.get(edge_key, 0)
                    neighbor_label = node_labels[neighbor]  # 使用最新标签
                    label_weights[neighbor_label] += weight
                
                if label_weights:
                    # 选择权重最大的标签
                    max_weight = max(label_weights.values())
                    best_labels = [label for label, weight in label_weights.items() 
                                 if weight == max_weight]
                    
                    # 确保best_labels不为空且处理平局
                    if best_labels:
                        if len(best_labels) == 1:
                            node_labels[node] = best_labels[0]
                        else:
                            # 可以选择最小标签或随机选择
                            node_labels[node] = min(best_labels)  # 或 np.random.choice(best_labels)
                # 如果没有邻居或权重为0，保持原标签不变
            
            # 检查收敛
            if old_labels == node_labels:
                break
        
        # 构建聚类结果
        clusters = defaultdict(set)
        for node, label in node_labels.items():
            clusters[label].add(node)
        
        return list(clusters.values())
    
    def spectral_clustering(self, n_clusters: int) -> List[Set[int]]:
        """
        谱聚类算法
        Args:
            n_clusters: 聚类数量
        Returns:
            聚类结果
        """
        # 构建权重矩阵
        node_list = self.nodes.tolist()
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        W = torch.zeros(self.num_nodes, self.num_nodes)
        for i in range(self.num_edges):
            u, v, w = self.edges[i, 0].item(), self.edges[i, 1].item(), self.edges[i, 2].item()
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            W[u_idx, v_idx] = w
            W[v_idx, u_idx] = w
        
        # 计算度矩阵
        D = torch.diag(W.sum(dim=1))
        
        # 计算拉普拉斯矩阵
        L = D - W
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # 选择前n_clusters个最小特征值对应的特征向量
        features = eigenvectors[:, :n_clusters]
        
        # 使用k-means聚类特征向量
        clusters = self._kmeans_clustering(features, n_clusters, node_list)
        
        return clusters
    
    def _kmeans_clustering(self, features, n_clusters, node_list):
        """简单的k-means聚类实现"""
        # 随机初始化聚类中心
        centers = features[torch.randperm(features.shape[0])[:n_clusters]]
        
        for _ in range(50):  # 最大迭代次数
            # 分配点到最近的中心
            distances = torch.cdist(features, centers)
            assignments = torch.argmin(distances, dim=1)
            
            # 更新聚类中心
            new_centers = torch.zeros_like(centers)
            for i in range(n_clusters):
                mask = assignments == i
                if mask.sum() > 0:
                    new_centers[i] = features[mask].mean(dim=0)
                else:
                    new_centers[i] = centers[i]
            
            # 检查收敛
            if torch.allclose(centers, new_centers, atol=1e-6):
                break
            
            centers = new_centers
        
        # 构建聚类结果
        clusters = [set() for _ in range(n_clusters)]
        for i, cluster_id in enumerate(assignments):
            clusters[cluster_id.item()].add(node_list[i])
        
        # 移除空聚类
        return [cluster for cluster in clusters if cluster]
    
    def visualize_clustering(self, clusters: List[Set[int]], title: str = "图聚类结果"):
        """
        可视化聚类结果
        Args:
            clusters: 聚类结果
            title: 图标题
        """
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加节点和边
        for i in range(self.num_edges):
            u, v, w = self.edges[i, 0].item(), self.edges[i, 1].item(), self.edges[i, 2].item()
            G.add_edge(u, v, weight=w)
        
        # 为每个聚类分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        node_colors = {}
        
        for i, cluster in enumerate(clusters):
            for node in cluster:
                node_colors[node] = colors[i]
        
        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制边，边的粗细表示权重
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')
        
        # 绘制节点
        for node in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=[node_colors.get(node, 'lightgray')],
                                 node_size=500, alpha=0.8)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 打印聚类信息
        print(f"\n聚类结果 ({len(clusters)} 个聚类):")
        for i, cluster in enumerate(clusters):
            print(f"聚类 {i+1}: {sorted(list(cluster))}")

# 使用示例
def example_usage():
    # 创建示例数据
    edges = torch.tensor([
        [1, 2, 0.8],
        [2, 3, 0.9],
        [3, 4, 0.7],
        [4, 5, 0.6],
        [5, 6, 0.8],
        [6, 7, 0.9],
        [1, 8, 0.3],
        [8, 9, 0.9],
        [9, 10, 0.8],
        [10, 11, 0.7],
        [2, 9, 0.2],
        [4, 10, 0.1]
    ])
    
    print("图聚类算法演示")
    print("=" * 50)
    
    # 初始化聚类器
    clusterer = GraphClustering(edges)
    
    # 方法1: 阈值聚类
    print("\n1. 阈值聚类 (阈值=0.6)")
    threshold_clusters = clusterer.threshold_clustering(0.6)
    print(f"聚类数量: {len(threshold_clusters)}")
    for i, cluster in enumerate(threshold_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 方法2: 层次聚类
    print("\n2. 层次聚类")
    hierarchical_results = clusterer.hierarchical_clustering()
    print("不同阈值下的聚类结果:")
    for threshold, clusters in hierarchical_results[:5]:  # 显示前5个阈值
        print(f"阈值 {threshold:.2f}: {len(clusters)} 个聚类")
        for i, cluster in enumerate(clusters):
            print(f"  聚类 {i+1}: {sorted(list(cluster))}")
        print()
    
    # 方法3: 模块度聚类
    print("3. 模块度聚类")
    modularity_clusters = clusterer.modularity_clustering()
    print(f"聚类数量: {len(modularity_clusters)}")
    for i, cluster in enumerate(modularity_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 方法4: Louvain算法
    print("\n4. Louvain算法")
    louvain_clusters = clusterer.louvain_clustering()
    print(f"聚类数量: {len(louvain_clusters)}")
    for i, cluster in enumerate(louvain_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 方法5: 标签传播算法
    print("\n5. 标签传播算法")
    lp_clusters = clusterer.label_propagation_clustering(seed=42)
    print(f"聚类数量: {len(lp_clusters)}")
    for i, cluster in enumerate(lp_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 方法6: 异步标签传播算法
    print("\n6. 异步标签传播算法")
    async_lp_clusters = clusterer.asynchronous_label_propagation(seed=42)
    print(f"聚类数量: {len(async_lp_clusters)}")
    for i, cluster in enumerate(async_lp_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 方法7: 谱聚类
    print("\n7. 谱聚类 (k=3)")
    spectral_clusters = clusterer.spectral_clustering(3)
    print(f"聚类数量: {len(spectral_clusters)}")
    for i, cluster in enumerate(spectral_clusters):
        print(f"聚类 {i+1}: {sorted(list(cluster))}")
    
    # 可视化（如果需要的话，取消注释）
    # clusterer.visualize_clustering(threshold_clusters, "阈值聚类结果")
    # clusterer.visualize_clustering(louvain_clusters, "Louvain聚类结果")
    # clusterer.visualize_clustering(lp_clusters, "标签传播聚类结果")

if __name__ == "__main__":
    example_usage()