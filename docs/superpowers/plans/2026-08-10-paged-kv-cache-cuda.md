# Paged KV Cache CUDA Append/Gather 执行计划

## 总目标

在不破坏现有CPU CI的前提下，完成可解释、可验证、可测量的分页KV Cache Append/Gather CUDA算子。

## Task 1：固定索引语义与CPU基线

- [ ] 实现纯CPU槽位到 `(block_id, block_offset)` 的映射函数。
- [ ] 增加跨块边界、越界、重复槽位和真实Qwen形状测试。
- [ ] 明确连续布局、dtype、设备与槽位约束。
- [ ] 验收：全量pytest与Ruff通过，CPU CI不依赖CUDA。

## Task 2：核验CUDA扩展工具链

- [ ] 核对运行环境中的PyTorch CUDA、CUDA_HOME、nvcc、g++和Ninja可用性。
- [ ] 确认所有新构建缓存和产物落在D盘项目目录。
- [ ] 如果需要安装新组件，先向用户说明组件、作用、大小和D盘路径，再安装。
- [ ] 验收：最小C++/CUDA扩展可以编译、加载并执行。

## Task 3：实现PyTorch参考与朴素CUDA Append

- [ ] 编写完整PyTorch参考实现。
- [ ] 编写C++绑定与输入校验。
- [ ] 编写朴素Append Kernel。
- [ ] 在float16、真实Qwen形状和随机slot下逐元素比对。
- [ ] 验收：正常、边界和非法输入测试全部通过。

## Task 4：实现Gather

- [ ] 编写PyTorch Gather参考实现。
- [ ] 编写朴素Gather Kernel。
- [ ] 覆盖顺序、乱序、跨块和边界槽位。
- [ ] 验收：Key和Value输出均与参考实现位级一致。

## Task 5：建立微基准

- [ ] 使用CUDA Event实现预热和正式计时。
- [ ] 覆盖不同Token数、KV头数、head_dim和block_size。
- [ ] 将checked输入校验与unchecked计时热路径分开。
- [ ] 报告Kernel P50/P95/P99、相邻版本加速比、累计加速比和有效带宽。
- [ ] 分离编译、数据准备、同步和Kernel执行时间。

## Task 6：建立V0朴素性能基线

- [ ] 实现一个线程负责一个Token、标量循环复制的V0版本。
- [ ] 将现有连续元素线程映射重新命名为V1合并访存版本。
- [ ] 确保V0和V1使用相同输入、Stream、编译参数和计时边界。
- [ ] PyTorch参考实现只参与逐元素正确性验证，不参与加速比计算。

## Task 7：逐项Kernel消融

- [ ] V0 baseline：一个线程负责一个Token。
- [ ] V1 coalesced：相邻线程访问连续head元素。
- [ ] 实现满足对齐条件时的16字节向量化路径。
- [ ] 保留非对齐与尾部标量路径。
- [ ] 单独调整CTA/Warp线程布局，不与向量化合并提交。
- [ ] 地址计算、Qwen主形状特化等其他优化分别建立独立变体。
- [ ] 每个变体报告相对前一版本和相对V0的收益，负收益如实保留。

## Task 8：原始结果与Nsight验证

- [ ] 使用Nsight Systems确认时间线和同步边界。
- [ ] 使用Nsight Compute采集DRAM吞吐、访存效率、Warp效率和Occupancy。
- [ ] 按run_id保存manifest、完整命令、输入分布、Kernel样本、请求时间线和服务日志。
- [ ] Kernel与端到端指标分别生成summary，不互相换算收益。
- [ ] 更新README、实验报告、每日进度与简历数字。

## Task 9：同版本vLLM端到端消融

- [ ] 以vanilla vLLM 0.23.0作为服务基线，固定模型revision、Attention Backend和启动参数。
- [ ] Prefix Cache关闭/开启实验只改变Cache开关，并复用同一组PreparedRequest。
- [ ] 自定义Kernel只有在同版本vLLM中完成单变量替换后，才允许报告端到端增量。
- [ ] 没有完成严格接入对照时，删除TPOT降低等自定义Kernel端到端结论。

## 当前开始点

下一次编码从Task 4的Gather正确性基线继续。Gather完成后先建立V0朴素Kernel，再将现有连续元素实现归类为V1合并访存，之后按V2向量化、V3线程布局、V4其他优化逐项推进。用户亲手实现有简历含金量的核心代码；Codex负责完整源码讲解、审阅、文档和自动化验证。
