# 非马尔可夫噪声：模拟与 SPP 张量网络

本仓库研究有限时长控制下的时间相关噪声，以及 SPP 的张量网络表示和误差缓解。当前结果均为合成数值模拟，尚无真机噪声学习或硬件验证。

## 当前状态

| 模块 | 已实现 | 入口 |
|---|---|---|
| `spp_tn` | 经典 Hermite HEOM 的层级 MPS、逐层 SPP 时间 MPS、PEA 与约束 ALS-PEC；使用 TensorCircuit-NG NumPy 后端 | [说明与 API](spp_tn/README.md) |
| `nmnoise` | 经典有色噪声的 dense／空间 MPS 轨迹、量子浴 HEOM 与显式浴对照、受限两时段 SPP 基准 | [模型与运行说明](docs/nmnoise-guide.md) |

**当前优先事项是 MPO 实际执行与 CPU／GPU 优化**，具体步骤见[待办](#待办todo)。现有传播器为压缩 RK4；尚无显式 MPO 执行路径、TDVP 或 GPU 路径。SPP 时间 MPS 和 HEOM 内部 MPS 压缩的是不同指标，不能与旧轨迹模拟的空间 MPS 混称。

## 快速运行

在代码仓库根目录（本地项目的 `Code/`）执行：

```bash
conda env create -f environment.yml
conda activate TensorCircuitNG
python -m pytest -q
python -m spp_tn.demo --output results/my_spp_run --convergence
```

已有环境可直接激活。输出目录应使用新名字；固定依赖统一维护在 [requirements.txt](requirements.txt)，也可在干净的 Python 3.11 环境中用 `python -m pip install -r requirements.txt` 安装。

TensorCircuit-NG 与 legacy TensorCircuit 共用导入名，需分环境安装。环境配置与历史验证见 [environments/README.md](environments/README.md)。2026-11-04 两套环境各通过 36 项项目测试；这不是本次重新测试的结果。核心求解无需 LaTeX，生成论文风格图表才需要 LaTeX 和字体。

## 结果与讨论入口

- [SPP 太大了，张量网络能帮上忙吗？](docs/spp-too-big.md)：问题说明、逐层构造及本次实验结果。
- [单比特 PEA／PEC 与收敛](results/spp_tn_20261104_final/README.md)：已知模型内的缓解与误差检查。
- [四比特单模 SPP 构造对照](results/spp_tn_ising4_mps_20261104/README.md)：紧凑表示与同层级参考核对；尚未建立扩规模收益或完整历史非负性。
- [报告、方案与数据索引](docs/reading-index.md)：其余讨论稿、HTML 方案、旧基准与文献核对记录。
- [复现与备份](docs/backup-and-reproduce.md)：结果归档和复现约定。

## 文稿与文献

文稿位于同级 [Overleaf/](../Overleaf/)，使用独立 Git 远端。文献在项目 [Zotero/](../Zotero/)；逐篇主题和阅读入口见 [Zotero/README.md](../Zotero/README.md)。Zotero 群组集合为 [Error-Mitigation → Non-Markov](https://www.zotero.org/groups/6729418/collections/FH728L4C)。这些同级目录是本地项目入口，不包含在代码仓库的独立克隆中。

## 待办（TODO）

### 当前优先：MPO 实际执行与 CPU／GPU 优化（2026-11-14）

下一阶段围绕 HEOM → SPP 的构造成本展开。首先实现 HEOM 生成元的矩阵乘积算符（MPO）表示及其对层级 MPS 的作用，再依据性能分析加入 CPU 优化或 GPU 批处理。PEA／PEC 保留为下游验证，当前不以扩展求逆算法为主线。

当前 `spp_tn/heom.py` 将生成元写成局域乘积算符之和，逐项作用、相加并压缩，使用 RK4 推进；尚无显式 MPO 类或 MPO–MPS 执行路径。`spp_tn/mps.py` 固定使用 NumPy 后端，且直接依赖 NumPy 数组，因此仅切换 TensorCircuit 的后端不能完成 GPU 迁移。本机为 ARM64，TensorCircuitNG 环境目前未安装 JAX、PyTorch 或 TensorFlow；GPU 路线须在目标设备上核验复杂数精度、QR/SVD 与批处理支持后确定。

- [ ] **MPO 正确性基线**：规定各节点的输入／输出指标和 `vec_F` 顺序，将现有 `terms()` 表示的生成元构造成 MPO，实现 MPO–MPS 作用与可控压缩。用小规模显式矩阵和原逐项实现对照；不自动做量子态归一化。生成元 MPO 不等于已经实现有限时间传播算符的 MPO，也不等于 TDVP。
- [ ] **接入 HEOM → SPP**：先保持 RK4、gate-after 探针及时间 MPS 构造不变，对比传播状态、SPP 联合张量误差和边缘概率，记录各类截断误差。后续是否更换积分器由误差和成本分析决定。
- [ ] **CPU 性能基线与优化**：分开测量算子作用、合并、QR/SVD 和探针传播；复用每段固定算子，评估收缩顺序、临时数组及独立探针批处理／并行。对并行工作数与 BLAS 线程数共同限额，避免线程过量；压缩顺序改变后重新验证误差。
- [ ] **GPU 候选路径**：将核心数组与运算改为可选后端，先验证 complex128 和分解运算，再评估同形状探针批处理。动态键维数、补零成本及主机／设备传输须纳入计时；后端和目标 GPU 尚未确定，不预设 GPU 一定更快。
- [ ] **验收实验**：保留已有单比特与四比特单模基线，增加噪声模式数、层级数和线路层数。在相同物理参数及达到相同目标误差的条件下，比较原逐项 CPU、新 MPO CPU 与可用的 GPU 路径；报告时间键、内部键、运行时间、峰值内存／显存和概率有效性。GPU 计时须同步，分别报告编译／预热与稳定运行成本。

完成标准是 MPO 传播正确、HEOM → SPP 端到端可运行，并获得可复现的精度与资源对照。是否获得加速，以及何种规模开始受益，由这些数据判断。

以下为后续研究方向。已完成配对序列的合成噪声数值扫描，尚未执行真机实验。

- [x] 实现可供对接的受限 4 比特 Ising 数值基准，完成 HEOM→SPP、末态计数学习、独立层/相关模型留出预测、L1 历史卷积求逆和 PEC 成本核验（2026-10-29；[报告](docs/spp-ising-benchmark.md)）。
- [ ] 接收合作者源码、参数、SPP 指标和优化操作集合，与本地基准逐项比较；本地自建基准不等于已验证其报告。
- [ ] 在扩展前明确多站点浴、更多时段与有限时长控制的范围；当前四历史模型无张量压缩或一般空间相关。器件拟合与硬件验证继续保留为未完成。

- [ ] 真机测量 GHZ 电路与从 `|0>` 出发的门电路终态，并测量相应的 X 乘积观测量期望值；比特数和具体测量协议仍需明确。
- [ ] 将真机 schedule 与波形输入数值模型，拟合环境/控制参数并比较硬件预测；HEOM 层级单独做数值收敛检查。
- [ ] 按 xiaoyang 学长建议，研究 Lindblad 与非马尔可夫模型差异是否可在真机上分辨；这是研究设想，尚非已证实的 non-Markovian effect。
- [ ] 在已有配对扫描基础上输入实验波形、校准控制/读出误差，完成参数不确定度下的实验选择与真机留出预测。
- [ ] 按 [SPP 新方案](docs/paired-pulse-proposal.md#合作者新方案学习时空-pauli-过程并用于预测与缓解2026-10-21)先确认多时刻对象与可用测量，完成一比特受控相关噪声正对照及可辨识性检查，再推进真机留出预测。
- [ ] 用现有 $d=0,2,4,8$ 的保真度检验时间齐次单指数基线（在保真度为正且对数误差可控时检查 $\log f_j$ 的线性），并比较准静态与有限相关时间模型的留出预测；非指数衰减本身不能证明“真记忆”或 CP 不可分性，见 [IBM 核对说明](docs/ibm-2607-25998-review.md)。
- [ ] 建立或选择噪声学习方案，研究浴相关函数/谱参数的可辨识性，以及学习误差对预测或缓解的影响。
- [ ] 选择具体器件/论文参数，统一谱归一化、温度和截止，拟合 BCF 并检查连续谱与 HEOM 层级收敛，再扩展到两比特 CR 控制。
- [ ] 旧量子浴路线的层级 MPS／TEMPO、可扩展局域观测量输出及更一般控制接口仍待实现；新 `spp_tn` 已有经典 Hermite HEOM 的层级 MPS，二者需区分。

## 维护约定与记录

- 区分已知模型数值验证、真机噪声学习和器件拟合，记录各自的完成范围。
- 每次运行保存参数、版本、源码快照及独立输出目录；分别检查时间步长、层级和张量截断误差。
- 区分时间相关性与 CP 不可分性，见[噪声模型层级](docs/pauli-noise-model-hierarchy.md)。

维护历史、已完成事项与环境迁移记录见 [maintenance-history.md](docs/maintenance-history.md)。2026-11-15 整理文档入口；未修改模拟实现或已有结果。
