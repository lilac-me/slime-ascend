## 权重转换指导

权重转换支支持两种方案：
1. 离线转换：hugginface 格式 -> torch dist 格式，也就是提供的样例脚本. **训练时候需要提供 huggingface 原始权重路径 + dist 权重路径.**

*Note :* 转换时候，指定的 --nproc-per-node 就是单节点上的 world_size，该值本质上就是 **pp_size**，那么在进行模型训练时候对应的 `--pipeline-model-parallel-size` 也就得与该值相等.

2. 在线转换：使用 Megatron-Bridge 方式，仅需传入 huggingface 原始权重地址，MBridge 会自动完成权重到 Megatron 格式的转换（整网训练该功能还有卡点，定位中...）