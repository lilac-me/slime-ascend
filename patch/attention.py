# 原始权重布局(列方向交织,按 KV group):
# ┌───────────────────────────────┬───────────────────────────────┐
# │  q1  q2  q3  q4   k1   v1     │   q5  q6  q7  q8   k2   v2    │
# │          group 0              │           group 1             │
# └───────────────────────────────┴───────────────────────────────┘
#                          ↓ column-parallel 切 4 份
# rank0: q1 q2 q3 │ rank1: q4 k1 v1 │ rank2: q5 q6 q7 │ rank3: q8 k2 v2

#                          ↓ Step 1: AG (all_gather_last_dim)
# 各 rank 都拿到:q1 q2 q3 q4 k1 v1 | q5 q6 q7 q8 k2 v2

#                          ↓ Step 2: idx = rank // 2,取自己那个 group
# rank 0,1:  q1 q2 q3 q4 k1 v1     (KV 冗余)
# rank 2,3:  q5 q6 q7 q8 k2 v2

#                          ↓ Step 3: 按 [4Q, 1K, 1V] 切
# Q = [q1,q2,q3,q4]   K = [k1]   V = [v1]    (rank 0,1)
# Q = [q5,q6,q7,q8]   K = [k2]   V = [v2]    (rank 2,3)

#                          ↓ Step 4: 按 rank % 2 再切 Q
# rank 0: Q=[q1,q2]  K=[k1]  V=[v1]
# rank 1: Q=[q3,q4]  K=[k1]  V=[v1]
# rank 2: Q=[q5,q6]  K=[k2]  V=[v2]
# rank 3: Q=[q7,q8]  K=[k2]  V=[v2]

#                          ↓ Step 5: core attention 内 repeat_interleave
# rank 0: Q=[q1,q2]  K=[k1,k1]  V=[v1,v1]    ← 正常跑 attention


# Attention class:
# """
# Derives `query`, `key` and `value` tensors from `hidden_states`.
# If `output_gate` is True, then also derives `gate` tensor.
# If `split_qkv=False`, then the unsplit mixed_qkv tensor is returned.
# """
# # For normal attention without groups, num_query_groups == num_attention_heads,
# # so these two will be the same (MHA)
# self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
# self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

# world_size = get_pg_size(self.pg_collection.tp)
# self.hidden_size_per_attention_head = self.query_projection_size // self.num_attention_heads
# if self.config.num_query_groups < world_size:
#     # When num_kv_heads < tp_size, each TP rank (post AG) initially produces
#     # activations for 1 kv_head and (num_q_heads / num_kv_heads) q_heads.
#     # We then pull out the appropriate (num_q_heads / tp_size) q_heads.
#     self.num_query_groups_per_partition = 1
#     self.num_attention_heads_per_partition = self.num_attention_heads // self.config.num_query_groups
# else:
#     self.num_query_groups_per_partition = self.config.num_query_groups // world_size
#     self.num_attention_heads_per_partition = self.num_attention_heads // world_size
# self.world_size = world_size

# if self.config.num_query_groups < world_size:
#     # TE throws an assertion error if num_kv_heads / num_query_groups
#     # is not divisible by TP size.
#     # TODO(rwaleffe/dnarayanan): Clean this up eventually.
#     tmp_config = copy.deepcopy(self.config)
#     tmp_config.num_query_groups = world_size
# else:
#     tmp_config = self.config

import copy
import torch
from torch import Tensor


# SelfAttention class:
def get_query_key_value_tensors(
    self,
    hidden_states: Tensor,
    key_value_states: Tensor | None = None,
    split_qkv: bool = False,
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, list[int]]
):
    mixed_qkv = apply_module(self.linear_qkv)(hidden_states)
    # two situations:
    # 1. num_query_groups > tp_size:
    # num_query_heads_per_group = (num_attention_heads // tp_size) // (num_query_groups // tp_size)
    # 2. num_query_groups <= tp_size:
    # num_query_heads_per_group = (num_attention_heads // num_query_groups) // 1
    # and the result is the same.
    num_query_heads_per_group = (
        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
    )

    num_qkv_heads_per_group = num_query_heads_per_group + 2
    
    assert self.config.num_query_groups is not None
    if self.config.num_query_groups < self.world_size:
        # Note that weights are interleaved in the following manner:
        # q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ...
        # When tp_size > num_kv_heads, we split "q1 q2 k1 v1" over multiple
        # ranks, so a rank does not have a clean partitioning of just the q_heads
        # it needs. Instead, we perform the following steps:
        # 1. Assemble the full "q1 q2 k1 v1 | q3 q4 k2 v2 | q5 q6 k3 v3 | ..."
        #    through an AG.
        # 2. Pull out the right slice (e.g., "q1 q2 k1 v1" or "q3 q4 k2 v2").
        # 3. Split q_heads (e.g., q1, q2), k_heads (e.g., k1), v_heads (e.g., v1).
        # 4. Further index into query to get only the q_heads that this rank is
        #    responsible for (e.g., q1).
        # The block of code below performs steps 1 and 2.
        mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
        idx = get_tensor_model_parallel_rank() // (
            self.world_size // self.config.num_query_groups
        )
        size = mixed_qkv.size()[-1] // self.config.num_query_groups
        mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

    # If no output gate: [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
    new_tensor_shape = mixed_qkv.size()[:-1] + (
        self.num_query_groups_per_partition,
        num_qkv_heads_per_group * self.hidden_size_per_attention_head,
    )
    mixed_qkv = mixed_qkv.view(*new_tensor_shape)

    # If no output gate: [sq, b, ng, (np/ng + 2) * hn]
    # --> [sq, b, ng, np/ng * hn], None, [sq, b, ng, hn], [sq, b, ng, hn]
    split_arg_list = [
        num_query_heads_per_group * self.hidden_size_per_attention_head,
        self.hidden_size_per_attention_head,
        self.hidden_size_per_attention_head,
    ]

    # Return unsplit mixed_qkv and split_arg_list
    if not split_qkv:
        return mixed_qkv, split_arg_list

    if SplitAlongDim is not None:
        (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
    else:
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
    
    # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
    query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

    if self.config.num_query_groups < self.world_size:
        # query above corresponds to (num_q_heads / num_kv_heads) q_heads.
        # Index appropriately into query to get (num_q_heads / tp_size) q_heads.
        # This is step 4 in the list of steps above.
        idx = get_tensor_model_parallel_rank() % (
            self.world_size // self.config.num_query_groups
        )
        size = self.num_attention_heads_per_partition // (
            self.world_size // self.config.num_query_groups
        )
        query = query[:, :, idx * size : (idx + 1) * size, :]
        
    return query, key, value