from typing import List
import torch
from torch import nn
from collections import defaultdict, deque
import math
import random
import torch.nn.functional as F
import numpy as np
from simdiff.fastcluster import FastSpectralClustering
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
class SimDiff(nn.Module):
    def __init__(self, cost=0.3, similarity_lower_bound=0.6, ratio_lower_bound=0.1,padding=-1,strategy=2, right: bool = True, bottom: bool =True, spatial: bool = True, temporal: bool = True, event_upper_bound=0.2):
        super(SimDiff, self).__init__()
        self.cost = cost
        self.similarity_lower_bound = similarity_lower_bound
        self.ratio_lower_bound = ratio_lower_bound
        self.event_upper_bound = event_upper_bound
        self.strategy=strategy
        self.right=right
        self.bottom=bottom
        self.spatial = spatial
        self.temporal = temporal
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

            image_token_pruning_start_index: int = self.image_token_start_index.item()
            image_token_pruning_length: int = (self.image_token_length - (self.original_length - q_len))

            last_layer_attention = self_attn_weights
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=(1,2))[0]
            last_layer_attention_avg_image = last_layer_attention_avg[image_token_pruning_start_index:image_token_pruning_start_index+image_token_pruning_length]

            pruning_ratio = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            # print('prune ratio: ', pruning_ratio)
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
            self.patch_type = self.patch_type.to(device)

            # prefill
            sparsity_upper_bound = self._compute_pruning_ratio(self.sparsity_list, self.cost)
            # print('merge ratio: ', sparsity_upper_bound)
            similarity_by_patch, token_index_by_patch = self.compute_similarity_and_token_index_by_patch(hidden_states, self.patch_type, self.patch_num) # only support bsz = 1

            
            frame_token_num = torch.sum(self.patch_type != TEXT_TOKEN).item()
            # merge_index_by_patch = torch.where(similarity_by_patch >= self.similarity_lower_bound)[1]
            # above_k_ratio = merge_index_by_patch.shape[0] / frame_token_num

            graph=[]

            event_token_ind = set()
            change_threshold = self.event_upper_bound

            ## 注释掉时间边
            for index,_ in enumerate(similarity_by_patch[0]):
                if similarity_by_patch[0][index] > -2 and similarity_by_patch[0][index]<change_threshold:
                    event_token_ind.add(token_index_by_patch[0,index].item())
                if self.temporal:
                    if similarity_by_patch[0][index]>=self.similarity_lower_bound:
                        graph.append(torch.tensor([token_index_by_patch[0,index-1],token_index_by_patch[0,index],similarity_by_patch[0][index].item()]))

            if self.spatial:
                right,bottom=self.compute_token_similarities_vectorized(
                    hidden_states[0,self.image_token_start_index:self.image_token_end_index+1,:],
                    self.n_frames,
                    self.height,
                    self.width
                    )
                
                for f in range(self.n_frames):
                    for h in range(self.height):
                        for w in range(self.width):
                            if right[f,h,w]>self.similarity_lower_bound:
                                graph.append(torch.tensor([f*self.patch_num+h*self.width+w+self.image_token_start_index,f*self.patch_num+h*self.width+w+1+self.image_token_start_index,right[f,h,w]]))

                for f in range(self.n_frames):
                    for h in range(self.height):
                        for w in range(self.width):
                            if bottom[f,h,w]>self.similarity_lower_bound:
                                graph.append(torch.tensor([f*self.patch_num+h*self.width+w+self.image_token_start_index,f*self.patch_num+(h+1)*self.width+w+self.image_token_start_index,bottom[f,h,w]]))
            
            self.finish_merging=True
            token_mask = torch.ones(hidden_states.shape[:-1], dtype=torch.bool, device=device)
            components_dfs = []
            from pdb import set_trace
            print('edges:',len(graph))
            if len(graph)>0:
                graph_tensor=torch.stack(graph)
                

                components_dfs,component_edges = find_connected_components_union_find(graph_tensor)
                # print('num_component:',len(components_dfs))
                # cal_coms={}
                # for i in components_dfs:
                #     if len(i) not in cal_coms:
                #         cal_coms[len(i)]=1
                #     else:
                #         cal_coms[len(i)]+=1
                # result,_=calculate_percentiles(dict(sorted(cal_coms.items())),percentiles=[90,99])
                # unique_node=0
                # temp_i=0
                
                # while temp_i <len(components_dfs):
                #     if len(components_dfs[temp_i])>=result['99th percentile']:
                #         components_dfs.pop(temp_i)
                #         component_edges.pop(temp_i)
                #     elif len(component_edges[temp_i])>0 and len(components_dfs[temp_i])<result['99th percentile'] and len(components_dfs[temp_i])>=result['90th percentile']:
                #         c=FastSpectralClustering(n_clusters=math.ceil(len(components_dfs[temp_i])/5))
                #         temp_component,_,_=c.fit_predict(component_edges[temp_i])
                #         for item in temp_component:
                #             if len(item)<=1:
                #                 continue
                #             components_dfs.append(item)
                #             component_edges.append([])
                #     else:
                #         temp_i+=1
                
                unique_node=sum([len(i)for i in components_dfs])
                #org_hidden_states, org_token_mask = self.merge_tokens_and_get_mask(hidden_states, similarity_by_patch, token_index_by_patch, merge_index_by_patch)
                if len(components_dfs)>0:
                    # above_k_number = unique_node-len(components_dfs)
                    # print(above_k_number)
                    # if above_k_number/ frame_token_num >= sparsity_upper_bound:
                    # sparsity_ratio: merge比例

                    sum_weight=torch.zeros(len(components_dfs))
                    per_component_weight=[]
                    if self.strategy==1:
                        for index in range(len(components_dfs)):
                            sum_weight[index]=sum(components_dfs[index])
                    else:
                        for index in range(len(components_dfs)):
                            if len(components_dfs)==0:
                                continue
                            # print(hidden_states[0][components_dfs[index]].shape)
                            # if hidden_states[0][components_dfs[index]].shape[0] > 1000:
                            #     # 如果大于1000，则随机取1000个向量来计算
                            #     idx = torch.randperm(hidden_states[0][components_dfs[index]].shape[0])[:1000]
                            #     components = hidden_states[0][components_dfs[index]][idx]
                            # else:
                            #     components = hidden_states[0][components_dfs[index]]

                            components = hidden_states[0][components_dfs[index]]
                            components_norm = F.normalize(components, p=2, dim=1)
                            cos_sim = torch.mm(components_norm, components_norm.T)
                            per_component_weight.append(cos_sim.mean(dim=-1))
                            if self.strategy==2:
                                cos_sim_sum=cos_sim.float().sum()/2/(len(cos_sim)-1)
                            elif self.strategy==3:
                                cos_sim_sum=(cos_sim.float().sum()-len(cos_sim))/(len(cos_sim)-1)/len(cos_sim)
                                # cos_sim_sum=(cos_sim.float().sum() - len(cos_sim))/(len(cos_sim)-1)/len(cos_sim)

                            assert not ((torch.isinf(cos_sim_sum) or torch.isnan(cos_sim_sum)))
                            sum_weight[index]=cos_sim_sum.item()
                    _,indices=torch.sort(sum_weight, descending=True)
                    # temp_index=0
                    # while temp_index<len(indices)and above_k_number/ frame_token_num >= sparsity_upper_bound:
                    #     temp_comp=components_dfs[indices[temp_index]]
                    #     above_k_number-=(len(temp_comp)-1)
                    #     temp_index+=1
                    # for i in range(temp_index):
                    #     components_dfs[indices[i]]=[]

                    # 正式开始merge
                    merge_token_num = 0
                    for index in indices:
                        merge_size = math.ceil(len(components_dfs[index]) * sparsity_upper_bound)
      
                        # 超额merge暂停
                        if (merge_token_num + merge_size) / frame_token_num > sparsity_upper_bound:
                            break

                        # 按内部相似度从高到底选择merge_size个索引
                        token_weight = per_component_weight[index]
                        _, token_indices = torch.sort(token_weight, descending=True)
                        for token_ind in token_indices[-merge_size:]:
                            if components_dfs[index][token_ind] in event_token_ind:
                                merge_size -= 1
                                continue
                            token_mask[0, components_dfs[index][token_ind]] = False
    


                        # _, token_indices = torch.sort(token_weight)
                        # merged_num = 0
                        # for token_ind in token_indices:
                            # if components_dfs[index][token_ind] in event_token_ind:
                            #     continue
                            # else:
                            # token_mask[0, components_dfs[index][token_ind]] = False
                            # merged_num += 1

                            # if merged_num == merge_size:
                            #     break
                            
                            
                        # # 随机选择merge_size个索引
                        # for token_ind in random.sample(components_dfs[index], merge_size):
                        #     token_mask[0, token_ind] = False

                        # 全merge到一个token
                        # for j in range(1,len(components_dfs[index])):
                        #     token_mask[0, components_dfs[index][j]] = False
                        #     hidden_states[0,components_dfs[index][0]]+=hidden_states[0,components_dfs[index][j]]
                        # if len(components_dfs[index])>1:
                        #     hidden_states[0,components_dfs[index][0]]/=(len(components_dfs[index])-1)

                        merge_token_num += merge_size

                    assert (merge_token_num / frame_token_num) == ((token_mask == False).sum().item() / frame_token_num)
                    merge_ratio = merge_token_num / frame_token_num
                    self.sparsity_list.append(merge_ratio)
                    self.finish_merging=True
                
            
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
