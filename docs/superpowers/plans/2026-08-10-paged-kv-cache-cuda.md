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
- [ ] 报告耗时、加速比和有效带宽。
- [ ] 分离编译、数据准备、同步和Kernel执行时间。

## Task 6：向量化与合并访存优化

- [ ] 实现满足对齐条件时的16字节向量化路径。
- [ ] 保留非对齐与尾部标量路径。
- [ ] 调整线程布局，使相邻线程访问连续head维度。
- [ ] 与朴素CUDA版本在相同输入上比较，不只与Python调用比较。

## Task 7：Nsight验证与成果记录

- [ ] 使用Nsight Systems确认时间线和同步边界。
- [ ] 使用Nsight Compute采集DRAM吞吐、访存效率、Warp效率和Occupancy。
- [ ] 保存命令、报告路径、硬件环境和关键截图或表格。
- [ ] 更新README、实验报告、每日进度与简历数字。

## 当前开始点

下一次编码从Task 1开始。用户亲手实现有简历含金量的索引语义与核心代码；Codex负责提供完整代码、逐段源码讲解、审阅、文档和自动化验证。
