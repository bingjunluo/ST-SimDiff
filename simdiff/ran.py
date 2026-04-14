from typing import List
import torch
from torch import nn
from collections import defaultdict, deque
import math
import torch.nn.functional as F
import numpy as np
from simdiff.fastcluster import FastSpectralClustering
import random
TEXT_TOKEN = -1
IGNORE_TOKEN = -2
def calculate_percentiles(freq_dict, percentiles=[25, 50, 75]):
    """
    计算字典中指定分位数对应的key
    
    Args:
        freq_dict: 字典，key为项目，value为出现次数
        percentiles: 要计算的分位数列表，默认为[25, 50, 75]
    
    Returns:
        dict: 包含各分位数及其对应key的字典
    """
    if not freq_dict:
        return {}
    
    # 方法1：将所有出现的项目展开成列表
    expanded_list = []
    for key, count in freq_dict.items():
        expanded_list.extend([key] * count)
    
    # 计算分位数
    result = {}
    for p in percentiles:
        percentile_value = np.percentile(expanded_list, p, method='nearest')
        result[f'{p}th percentile'] = percentile_value
    
    return result, expanded_list
def find_connected_components_dfs(edges):
    """
    使用深度优先搜索（DFS）找出图中的所有连通分量
    
    参数:
    edges: torch.Tensor, 形状为 [n, 2]，每行表示一条边连接的两个节点
    
    返回值:
    list of lists: 每个子列表包含一个连通分量中的所有节点
    """
    # 构建邻接表
    graph = defaultdict(list)
    nodes = set()
    
    # 将edges转为numpy以便处理
    if isinstance(edges, torch.Tensor):
        edges = edges.numpy()
    
    # 创建邻接表并获取所有节点
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # 无向图
        nodes.add(u)
        nodes.add(v)
    
    # DFS函数
    def dfs(node, component, visited):
        visited[node] = True
        component.append(node)
        
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, component, visited)
    
    # 初始化访问状态
    visited = {node: False for node in nodes}
    components = []
    
    # 遍历所有节点，找出连通分量
    for node in nodes:
        if not visited[node]:
            component = []
            dfs(node, component, visited)
            components.append(sorted(component))  # 排序以保持一致性
    
    # 按照第一个元素排序整个结果
    components.sort(key=lambda x: x[0] if x else float('inf'))
    return components

def split_lists_by_value(data, cos_sim_func, alpha, beta, theta,strategy,hidden):
    """
    对二维列表进行基于value的智能拆分
    
    参数:
    data: 二维列表
    cos_sim_func: 计算列表value的函数，接受一个列表作为参数，返回一个数值
    alpha: 长度阈值，超过此长度的列表需要拆分
    beta: 拆分后每个子列表的最大长度
    theta: value阈值，低于此值的列表需要拆分
    
    返回:
    tuple: (处理后的二维列表, 对应的value列表)
    """
    if beta <= 0:
        raise ValueError("beta必须大于0")
    
    result_lists = []
    result_values = []
    for sublist in data:
        # 计算当前列表的value
        current_value = cos_sim_func(strategy,hidden,sublist)
        
        # 判断是否需要拆分：长度超过alpha 或者 value低于theta
        if len(sublist) > alpha and current_value < theta:
            # 需要拆分的列表
            split_sublists = random_split(sublist, beta)
            
            # 计算拆分后每个子列表的value
            for split_sublist in split_sublists:
                split_value = cos_sim_func(strategy,hidden,split_sublist)
                result_lists.append(split_sublist)
                result_values.append(split_value)
        else:
            # 不需要拆分的列表直接添加
            result_lists.append(sublist)
            result_values.append(current_value)
    
    return result_lists, result_values

def random_split(lst, max_size):
    """
    将列表随机拆分为多个子列表，每个子列表长度不超过max_size
    
    参数:
    lst: 要拆分的列表
    max_size: 每个子列表的最大长度
    
    返回:
    拆分后的子列表组成的列表
    """
    if not lst:
        return []
    
    # 创建列表副本并随机打乱
    shuffled_list = lst.copy()
    random.shuffle(shuffled_list)
    
    # 按max_size大小进行拆分
    result = []
    for i in range(0, len(shuffled_list), max_size):
        result.append(shuffled_list[i:i + max_size])
    
    return result
def find_connected_components_union_find(edges):
    """
    使用并查集（Union-Find）找出图中的所有连通分量
    
    参数:
    edges: torch.Tensor, 形状为 [n, 2]，每行表示一条边连接的两个节点
    
    返回值:
    tuple: (components, component_edges)
        - components: list of lists, 每个子列表包含一个连通分量中的所有节点
        - component_edges: list of lists, 每个子列表包含一个连通分量中的所有边
    """
    # 将edges转为numpy以便处理
    if isinstance(edges, torch.Tensor):
        edges_array = edges.numpy()
    else:
        edges_array = edges
    
    # 获取所有节点
    nodes = set()
    for u, v ,_ in edges_array:
        nodes.add(u)
        nodes.add(v)
    
    # 初始化并查集
    parent = {node: node for node in nodes}
    
    # 查找函数，带路径压缩
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    # 合并函数
    def union(x, y):
        parent[find(x)] = find(y)
    
    # 合并所有边连接的节点
    for u, v,_ in edges_array:
        union(u, v)
    
    # 收集连通分量的节点
    components_dict = defaultdict(list)
    for node in nodes:
        components_dict[find(node)].append(int(node))
    
    # 收集连通分量的边
    component_edges_dict = defaultdict(list)
    for i, (u, v,k) in enumerate(edges_array):
        # 找到这条边所属的连通分量（通过任一端点的根节点确定）
        root = find(u)
        component_edges_dict[root].append([u, v,k])
    
    # 转换为列表并排序
    components = [sorted(comp) for comp in components_dict.values()]
    components.sort(key=lambda x: x[0] if x else float('inf'))
    
    # 获取对应的边，保持与components相同的顺序
    component_edges = []
    for comp in components:
        if comp:  # 确保连通分量不为空
            root = find(comp[0])  # 获取该连通分量的根节点
            comp_edges = component_edges_dict[root]
            # 可选择按边进行排序
            comp_edges.sort(key=lambda x: (x[0], x[1]))
            component_edges.append(comp_edges)
        else:
            component_edges.append([])
    
    return components, component_edges
def cos_sim_func(strategy,hidden_states,component):
    if strategy==1:
        return sum(component)
    else:
       
        if hidden_states[0][component].shape[0] > 500:
            # 如果大于1000，则随机取1000个向量来计算
            idx = torch.randperm(hidden_states[0][component].shape[0])[:500]
            components = hidden_states[0][component][idx]
        else:
            components = hidden_states[0][component]
        cos_sim=F.cosine_similarity(components.unsqueeze(1),components.unsqueeze(0),dim=-1)
        if strategy==2:
            cos_sim_sum=cos_sim.float().sum()/2/(len(cos_sim)-1)
        elif strategy==3:
            cos_sim_sum=(cos_sim.float().sum()-len(cos_sim))/2/(len(cos_sim)-1)/len(cos_sim)
        return cos_sim_sum.item()
class SimDiff(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1,padding=-1,strategy=2, right: bool = True, bottom: bool =True,):
        super(SimDiff, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound
        self.strategy=strategy
        self.right=right
        self.bottom=bottom
        self.padding=padding

    def prepare(
        self,
        patch_type: torch.Tensor,
        patch_num: int,
        image_token_start_index: torch.Tensor,
        image_token_end_index: torch.Tensor,
        image_token_length: torch.Tensor,
        original_length: int,
        finish_merging: bool = False,
        finish_pruning: bool = False,
        height: int = None,
        width: int = None,
        sparsity_list: List[float] = None,
    ):
        self.patch_type = patch_type
        self.patch_num = patch_num
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length
        self.finish_merging = finish_merging
        self.finish_pruning = finish_pruning
        if height:
            self.height=height
            self.width=width
        else:
            self.width=math.ceil(math.sqrt(patch_num))
            self.height=self.width-1
        self.n_frames=image_token_length // patch_num
        if sparsity_list is None:
            self.sparsity_list = []
        else:
            self.sparsity_list = sparsity_list

    def forward(
        self, hidden_states, position_embeddings, attention_mask, self_attn_weights=None
    ):
        """
        This is the forward method of the SimDiff class.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
            self_attn_weights (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).

        Returns:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            position_embeddings (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): A tensor of shape (batch_size, sequence_length, sequence_length).
        """
        bsz, q_len, hidden_size = hidden_states.size()
        device = hidden_states.device    
        # pruning
        if q_len >1 and self.finish_merging == True and self.finish_pruning == False:
        # if q_len >1 and self.finish_pruning == False:

            image_token_pruning_start_index: int = self.image_token_start_index.item()
            image_token_pruning_length: int = (self.image_token_length - (self.original_length - q_len))

            last_layer_attention = self_attn_weights
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=(1,2))[0]
            last_layer_attention_avg_image = last_layer_attention_avg[image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length]

            pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            top_attention_rank_index = (
                last_layer_attention_avg_image.topk(
                    round(image_token_pruning_length * (1 - pruning_ratio))
                ).indices
                + image_token_pruning_start_index
            )

            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_pruning_start_index, device=device),
                    top_attention_rank_index,
                    torch.arange(
                        image_token_pruning_start_index + image_token_pruning_length,
                        q_len,
                        device=device,
                    ),
                )
            )
            keep_indexs = keep_indexs.sort().values

            hidden_states = hidden_states[:,keep_indexs,:] 
            print('prune',q_len-hidden_states.shape[1])
            position_embeddings = self.position_embedding_handler_at_pruning(position_embeddings, keep_indexs)
            if attention_mask != None:
                attention_mask = attention_mask[:,:,keep_indexs,:][:,:,:,keep_indexs]
            self.finish_pruning = True
        sims=[]
        # merging
        if q_len >1 and (not self.finish_merging):
            # align devices
            token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            def random_select(mask,cost):
                n=mask.shape[0]
                m=int(10*(1-cost))
                for start_idx in range(0, n, 10):
                    end_idx = min(start_idx + 10, n)
                    group_size = end_idx - start_idx
                    
                    # 如果当前组的大小小于m，则选择所有元素
                    actual_m = min(m, group_size)
                    
                    # 当前组的索引
                    group_indices = list(range(start_idx, end_idx))
                    
                    # 随机选择m个索引
                    selected_group_indices = random.sample(group_indices, actual_m)
                    mask[selected_group_indices]=0
                return mask
            self.sparsity_list.append(1-self.cost)
            self.finish_merging=True
            token_mask[0,self.image_token_start_index:self.image_token_end_index+1]=random_select(token_mask[0,self.image_token_start_index:self.image_token_end_index+1],self.cost)
            self.patch_type = self.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, hidden_size)
            print('merge',q_len-hidden_states.shape[1])
            position_embeddings = self.position_embedding_handler_at_merging(position_embeddings, token_mask)
            if attention_mask is not None:
                attention_mask = attention_mask[:,:,token_mask[0],:][:,:,:,token_mask[0]]

        return hidden_states, position_embeddings, attention_mask
    def compute_token_similarities_vectorized(self,tensor, frames, height, width):
        """
        使用向量化操作更高效地计算二维tensor中每个位置的token与其右侧和下侧token的余弦相似度
        
        参数与返回值同上
        """

        # 将tensor重塑为[frames, height, width, embedding_dim]
        reshaped_tensor = tensor.reshape(frames,height,width, -1)
        
        # 初始化结果张量
        right_similarities = torch.zeros(frames, height, width, device=tensor.device)
        bottom_similarities = torch.zeros(frames, height, width, device=tensor.device)
        
        # 计算每个位置与右侧token的余弦相似度（向量化）
        # 正则化每个token以计算余弦相似度
        normalized_tokens = F.normalize(reshaped_tensor, p=2, dim=-1)
        
        # 右侧相似度（对于每一行，计算相邻tokens的点积）
        if self.right:
            for f in range(frames):
                for h in range(height):
                    # 当前行所有tokens
                    current_row = normalized_tokens[f, h, :-1]  # 除了最后一列
                    next_col = normalized_tokens[f, h, 1:]      # 除了第一列
                    
                    # 计算相邻tokens的点积（已经正则化，所以点积等于余弦相似度）
                    similarities = torch.sum(current_row * next_col, dim=-1)
                    right_similarities[f, h, :-1] = similarities
        
        # 下侧相似度（对于每一列，计算相邻tokens的点积）
        if self.bottom:
            for f in range(frames):
                for w in range(width):
                    # 当前列所有tokens
                    current_col = normalized_tokens[f, :-1, w]  # 除了最后一行
                    next_row = normalized_tokens[f, 1:, w]      # 除了第一行
                    
                    # 计算相邻tokens的点积
                    similarities = torch.sum(current_col * next_row, dim=-1)
                    bottom_similarities[f, :-1, w] = similarities
        
        return right_similarities, bottom_similarities
    def position_embedding_handler_at_pruning(self, position_embeddings, keep_indexs):
        if type(position_embeddings) == list:
            assert len(position_embeddings) == 2
            if position_embeddings[0].ndim == 4:
                position_embeddings[0] = position_embeddings[0][:,:,keep_indexs,:]
                position_embeddings[1] = position_embeddings[1][:,:,keep_indexs,:]
            else:
                position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
                position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
        elif type(position_embeddings) == torch.Tensor:
            if position_embeddings.ndim == 2:
                position_embeddings = position_embeddings[:,keep_indexs]
            else:
                raise NotImplementedError("Only support 2D position embeddings")
        else:
            raise NotImplementedError("Only support list or tensor for position embeddings")
        return position_embeddings
    
    
    def position_embedding_handler_at_merging(self, position_embeddings, token_mask):
        if type(position_embeddings) == list:
            # (cos, sin)
            assert len(position_embeddings) == 2
            if position_embeddings[0].ndim == 4:
                position_embeddings[0] = position_embeddings[0][:,:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,:,token_mask[0],:]
            else:
                position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
                position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
        elif type(position_embeddings) == torch.Tensor:
            if position_embeddings.ndim == 2:
                position_embeddings = position_embeddings[:,token_mask[0]]
            else:
                raise NotImplementedError("Only support 2D position embeddings")
        else:
            raise NotImplementedError("Only support list or tensor for position embeddings")
        return position_embeddings
    @staticmethod
    def compute_similarity_and_token_index_by_patch(hidden_states, token_patch_type, patch_num):
        """
        Compute the similarity between consecutive tokens of the same patch type and record the token index.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size).
            token_patch_type (torch.Tensor): A tensor indicating the patch type of each token in the sequence.
            patch_num (int): The total number of patches of one image in the model.

        Returns:
            similarity_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the cosine similarity between consecutive tokens of the
                                                same patch type. Tokens from different patches are set to -2.
            token_index_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the token index corresponding to the new order after
                                                sorting by patch type.

        """

        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        assert bsz == 1, "Only support batch size 1"

        token_index_by_patch = []
        similarity_by_patch = []

        token_patch_type_by_patch, token_index_by_patch = torch.where(
            token_patch_type == torch.arange(patch_num, device=device)[:, None]
        )

        # noqa: reshape to batch size = 1, with shape (batch_size, q_len),
        token_patch_type_by_patch = token_patch_type_by_patch[None, :]
        token_index_by_patch = token_index_by_patch[None, :]

        similarity_by_patch = cosine_similarity(
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, :-1], :
            ],
            hidden_states[
                torch.arange(bsz, device=device), token_index_by_patch[:, 1:], :
            ],
        )

        similarity_by_patch[token_patch_type_by_patch[:, :-1] != token_patch_type_by_patch[:, 1:]] = -2
        # print(similarity_by_patch.shape)
        similarity_by_patch = torch.cat(
            (
                torch.full(
                    size=(bsz, 1),
                    fill_value=IGNORE_TOKEN,
                    dtype=hidden_states.dtype,
                    device=device,
                ),
                similarity_by_patch,
            ),
            dim=1,
        )
        # print(similarity_by_patch.shape)
        assert similarity_by_patch.shape[1] == token_index_by_patch.shape[1]
        return similarity_by_patch, token_index_by_patch

    @staticmethod
    def merge_tokens_and_get_mask(hidden_states: torch.Tensor, similarity_by_patch, token_index_by_patch, merge_index_by_patch):
        """
        Merge tokens and get a mask indicating which tokens to keep.

        Args:
            hidden_states (torch.Tensor): A tensor of shape (batch_size, sequence_length, hidden_size)
            similarity_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the cosine similarity between consecutive tokens of the
                                                same patch type.
            token_index_by_patch (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing
                                                the token indices corresponding to the new order after
                                                sorting by patch type.
            merge_index_by_patch (torch.Tensor): A tensor containing the indices of tokens to be merged, in the patch_type order.

        Returns:
            hidden_states (torch.Tensor): A tensor containing the hidden states of the tokens after merging.
            keep_mask (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length) indicating
                                    which tokens in the original sequence should be kept after merging.
        """
        device = hidden_states.device
        if merge_index_by_patch.shape[0] == 0:
            keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            return hidden_states, keep_mask
        bsz, q_len, _ = hidden_states.size()
        bsz_index = torch.arange(bsz, device=hidden_states.device)[:, None]
        merge_mask_by_patch: torch.LongTensor = torch.zeros(
            bsz,
            similarity_by_patch.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        merge_mask_by_patch[bsz_index, merge_index_by_patch] = 1
        last_merge_token_by_patch = find_contigious_latter_index(merge_mask_by_patch)

        keep_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
        keep_mask[bsz_index, token_index_by_patch[bsz_index, merge_index_by_patch]] = False

        # noqa: batch size = 1
        unique_merge_nums = torch.sort(torch.unique(last_merge_token_by_patch.to(torch.long))).values
        unique_merge_nums = (unique_merge_nums[1:] if (unique_merge_nums[0] == 0).item() else unique_merge_nums)

        merge_num_indices, token_merge_index_in_patch = torch.where(
            last_merge_token_by_patch == unique_merge_nums[:, None]
        )

        merge_nums = unique_merge_nums[merge_num_indices]
        token_merge_start_index_in_patch = token_merge_index_in_patch - merge_nums
        token_merge_member_start_index_in_patch = torch.repeat_interleave(token_merge_start_index_in_patch, merge_nums)

        merge_member_length = torch.sum(merge_nums)
        merge_member_contigious_sequence = torch.arange(1, merge_member_length + 1, device = device)

        merge_nums_cumulative_counts = torch.cumsum(merge_nums, dim=0)
        merge_nums_start = torch.cat((torch.tensor([0], device = device), merge_nums_cumulative_counts[:-1]))

        contigious_sequence_by_merge_nums = merge_member_contigious_sequence - torch.repeat_interleave(merge_nums_start, merge_nums)

        token_merge_member_index_in_patch = token_merge_member_start_index_in_patch + contigious_sequence_by_merge_nums

        # noqa: this function may have numerical instability
        hidden_states.index_add_(
            dim = 1,
            index = token_index_by_patch[0, token_merge_member_start_index_in_patch],
            source = hidden_states[
                bsz_index,
                token_index_by_patch[bsz_index, token_merge_member_index_in_patch],
            ]
        )  

        # divide to get average
        hidden_states[
            bsz_index,
            token_index_by_patch[bsz_index, token_merge_start_index_in_patch],
        ] /= (merge_nums[None, :, None] + 1)

        return hidden_states, keep_mask
    
    @staticmethod
    def _compute_pruning_ratio(sparsity_list, cost, num_layers = 28):
        """
        Args:
            sparsity_list (list): A list containing the sparsity values of the model's first few layers.
            cost (float): The total computation budget given by the user.
            num_layers (int, optional): The number of layers in the model. 

        Returns:
            float: the required sparsity for the next layer to achieve the given cost
        """
        list_length = len(sparsity_list)
        s = 1
        total_calcution =0
        for i in range(list_length):
            s *= (1 - sparsity_list[i])
            total_calcution += s
        remain_calcution = num_layers * cost - total_calcution
        if remain_calcution < 0:
            raise ValueError("The cost is too small")
        if remain_calcution/((num_layers-list_length)*s) > 1:
            return 0
        return 1 - (remain_calcution/((num_layers-list_length)*s))    

def cosine_similarity(mat1, mat2):
    dot_product = torch.sum(mat1*mat2, dim=-1)
    norm_vec1 = torch.norm(mat1, dim=-1)
    norm_vec2 = torch.norm(mat2, dim=-1)
    return dot_product / (norm_vec1 * norm_vec2)

def find_contigious_latter_index(index_tensor: torch.LongTensor) -> torch.Tensor:
    """
    Args:
        index_tensor (torch.LongTensor): A binary tensor containing sequences of ones and zeros.

    Returns:
        torch.Tensor: A tensor where each contiguous sequence of ones in the input tensor
                    is replaced by zeros, except for the last element of each sequence,
                    which is replaced by the length of that sequence.

    Example:
        Input:  torch.tensor([0, 1, 1, 1, 0, 0, 1, 1])
        Output: torch.tensor([0, 0, 0, 3, 0, 0, 0, 2])
    """
    bsz, n = index_tensor.shape
    t_prev = torch.cat([torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device), index_tensor[:, :-1]], dim=1)
    t_next = torch.cat([index_tensor[:, 1:], torch.zeros((bsz, 1), dtype=index_tensor.dtype, device=index_tensor.device)], dim=1)

    # Identify the starts and ends of runs of ones
    run_starts = (index_tensor == 1) & (t_prev == 0)
    run_ends = (index_tensor == 1) & (t_next == 0)

    start_indices = torch.nonzero(run_starts, as_tuple=True)
    end_indices = torch.nonzero(run_ends, as_tuple=True)
    run_lengths = (end_indices[1] - start_indices[1] + 1).to(index_tensor.dtype)

    output = torch.zeros_like(index_tensor, dtype=index_tensor.dtype)
    output[end_indices[0], end_indices[1]] = run_lengths

    return output
