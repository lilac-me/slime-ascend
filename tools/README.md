## 权重转换指导

权重转换支支持两种方案：
1. 离线转换：hugginface 格式 -> torch dist 格式，也就是提供的样例脚本. **训练时候需要提供 huggingface 原始权重路径 + dist 权重路径.**

*Note :* 转换时候，指定的 --nproc-per-node 就是单节点上的 world_size，该值本质上就是 **pp_size**

离线转换加载权重本质是利用 checkpoint 的 shared_state_dict 机制，存在 `local_shape`, `global_shape` 以及 `axis_fragmentation` 三个 shape 元组，关系为：

$$
local\_shape = global\_shape\ /\ axis\_fragmentation
$$

其中，
*global_shape*: 模型未做 tp、ep 切分前的 shape
*local_shape*: 模型做完 tp、ep 切分后的 shape
*axis_framentation*: 切片 size （与 tp、 etp 相关）


2. 在线转换：使用 Megatron-Bridge 方式，仅需传入 huggingface 原始权重地址，MBridge 会自动完成权重到 Megatron 格式的转换（整网训练该功能还有卡点，定位中...）