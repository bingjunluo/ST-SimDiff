from typing import List
import torch
from torch import nn
from collections import defaultdict, deque
import math
import torch.nn.functional as F
TEXT_TOKEN = -1
IGNORE_TOKEN = -2
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


def find_connected_components_union_find(edges):
    """
    使用并查集（Union-Find）找出图中的所有连通分量
    
    参数:
    edges: torch.Tensor, 形状为 [n, 2]，每行表示一条边连接的两个节点
    
    返回值:
    list of lists: 每个子列表包含一个连通分量中的所有节点
    """
    # 将edges转为numpy以便处理
    if isinstance(edges, torch.Tensor):
        edges = edges.numpy()
    
    # 获取所有节点
    nodes = set()
    for u, v in edges:
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
    for u, v in edges:
        union(u, v)
    
    # 收集连通分量
    components_dict = defaultdict(list)
    for node in nodes:
        components_dict[find(node)].append(node)
    
    # 转换为列表并排序
    components = [sorted(comp) for comp in components_dict.values()]
    components.sort(key=lambda x: x[0] if x else float('inf'))
    
    return components
def compute_token_similarities_vectorized(tensor, frames, height, width,interval):
    """
    使用向量化操作更高效地计算二维tensor中每个位置的token与其右侧和下侧token的余弦相似度
    
    参数与返回值同上
    """

    # 将tensor重塑为[frames, height, width, embedding_dim]
    reshaped_tensor = tensor.reshape(frames,height,width, -1)
    total_tokens = height * width * frames
    tokens_per_frame = height * width
    
    # 初始化相似度tensor
    right_similarity = torch.full((total_tokens,), -1.0, dtype=tensor.dtype, device=tensor.device)
    down_similarity = torch.full((total_tokens,), -1.0, dtype=tensor.dtype, device=tensor.device)

    valid_frames = frames - interval
    # 计算每个位置与右侧token的余弦相似度（向量化）
    # 正则化每个token以计算余弦相似度
    for frame in range(valid_frames):
        # 当前帧的token索引范围
        current_start = frame * tokens_per_frame
        current_end = (frame + 1) * tokens_per_frame
        
        # 目标帧的token索引范围
        target_start = (frame + interval) * tokens_per_frame
        target_end = (frame + interval + 1) * tokens_per_frame
        
        # 获取当前帧和目标帧的tokens
        current_tokens = tensor[current_start:current_end-1]  # shape: (tokens_per_frame, feature_dim)
        target_tokens = tensor[target_start+1:target_end]     # shape: (tokens_per_frame, feature_dim)
        
        # 计算余弦相似度（批量计算）
        right = F.cosine_similarity(current_tokens, target_tokens, dim=1)
        current_tokens = tensor[current_start:current_end-width]  # shape: (tokens_per_frame, feature_dim)
        target_tokens = tensor[target_start+width:target_end]     # shape: (tokens_per_frame, feature_dim)
        down=F.cosine_similarity(current_tokens, target_tokens, dim=1)
        # 由于在这个问题设定中，"右边"和"下边"都是指向同一个间隔帧的对应位置
        # 所以右边和下边的相似度实际上是相同的
        right_similarity[current_start:current_end-1] = right
        down_similarity[current_start:current_end-width] = down
    
    return right_similarity, down_similarity
class SimDiff(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1):
        super(SimDiff, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound

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
            position_embeddings[0] = position_embeddings[0][:,keep_indexs,:]
            position_embeddings[1] = position_embeddings[1][:,keep_indexs,:]
            if attention_mask != None:
                attention_mask = attention_mask[:,:,keep_indexs,:][:,:,:,keep_indexs]
            self.finish_pruning = True
        sims=[]
        # merging
        if q_len >1 and (not self.finish_merging):
            # align devices
            self.patch_type = self.patch_type.to(device)

            # prefill
            sparsity_upper_bound = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            similarity_by_patch, token_index_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type, self.patch_num) # only support bsz = 1

            
            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item()
            merge_index_by_patch = torch.where(similarity_by_patch >= self.similarity_lower_bound)[1]
            above_k_ratio = merge_index_by_patch.shape[0] / frame_token_num

                
            graph=[]
            for index,_ in enumerate(similarity_by_patch[0]):
                if similarity_by_patch[0][index]<0:
                    continue
                if similarity_by_patch[0][index]>=self.similarity_lower_bound:
                    graph.append(torch.tensor([token_index_by_patch[0,index-1],token_index_by_patch[0,index]]))
                    sims.append(similarity_by_patch[0][index].item())
                
            

            if not self.finish_merging:
                interval=1
                right,bottom=compute_token_similarities_vectorized(
                    hidden_states[0,self.image_token_start_index:self.image_token_end_index+1,:],
                    self.n_frames,
                    self.height,
                    self.width,
                    interval
                    )
                for index,i in enumerate(right):
                    if right[index]>self.similarity_lower_bound:
                        graph.append(torch.tensor([self.image_token_start_index+index,self.patch_num*interval+self.image_token_start_index+1]))
                        sims.append(right[index])
                for index,i in enumerate(bottom):
                    if bottom[index]>self.similarity_lower_bound:
                        graph.append(torch.tensor([self.image_token_start_index+index,self.patch_num*interval+self.image_token_start_index+self.width]))
                        sims.append(bottom[index])
                
                self.finish_merging=True
                # 下侧相似度（对于每一列，计算相邻tokens的点积）
            token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            components_dfs = []
            if len(graph)>0:
                graph_tensor=torch.stack(graph)
                

                components_dfs = find_connected_components_dfs(graph_tensor)
                #org_hidden_states, org_token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
                if len(components_dfs)>0:
                    unique_node=torch.flatten(graph_tensor)
                    unique_node=torch.unique(unique_node)
                    above_k_number = (unique_node.shape[0]-len(components_dfs)) 
                    from pdb import set_trace
                    
                    if (above_k_number/ frame_token_num) >= sparsity_upper_bound:
                        
                        sum_weight=torch.zeros(len(components_dfs))
                        for index in range(len(components_dfs)):
                            cos_sim=F.cosine_similarity(hidden_states[0][components_dfs[index]],hidden_states[0][components_dfs[index]])
                            cos_sim_sum=cos_sim.sum()/2/(len(cos_sim)-1)
                            sum_weight[index]=cos_sim_sum.item()
                            
                        _,indices=torch.sort(sum_weight,descending=True)
                        temp_index=0

                        while temp_index<len(indices)and (above_k_number/ frame_token_num) >= sparsity_upper_bound:
                            temp_comp=components_dfs[indices[temp_index]]
                            above_k_number-=(len(temp_comp)-1)
                            temp_index+=1

                        for i in range(temp_index):
                            components_dfs[indices[i]]=[]
                        s=0
                        for index,_ in enumerate(components_dfs):
                            s+=len(components_dfs[index])
                            for j in range(1,len(components_dfs[index])):
                                token_mask[0, components_dfs[index][j]] = False
  
                        self.finish_merging=True
                        self.finish_pruning=True

                    else:
                        self.sparsity_list.append(above_k_number/frame_token_num)
                        for index,_ in enumerate(components_dfs):
                            for j in range(1,len(components_dfs[index])):
                                token_mask[0, components_dfs[index][j]] = False
                        self.finish_merging = True
                

                
                
            # diff=[]
            # sims=[]
            # for index,_ in enumerate(similarity_by_patch[0]):
            #     if similarity_by_patch[0,index]>=self.similarity_lower_bound:
            #         sims.append(token_index_by_patch[0][index])
            # for index,_ in enumerate(org_token_mask[0]):
                
            #     if org_token_mask[0,index]!=token_mask[0,index]:
            #         diff.append(index)
            # assert (org_token_mask==token_mask).all()
            # assert (org_hidden_states==hidden_states).all()
            
            # here only bsz=1
            # update patch type
            self.patch_type = self.patch_type.to(device)[token_mask].reshape(bsz, -1)
            hidden_states = hidden_states[token_mask, :].reshape(bsz, -1, hidden_size)
            position_embeddings[0] = position_embeddings[0][:,token_mask[0],:]
            position_embeddings[1] = position_embeddings[1][:,token_mask[0],:]
            if attention_mask is not None:
                attention_mask = attention_mask[:,:,token_mask[0],:][:,:,:,token_mask[0]]

        return hidden_states, position_embeddings, attention_mask

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
