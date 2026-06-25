# QKD Dual-Capacity Routing Simulation

考虑 QKD 密钥容量约束的经典-量子融合通信网路由阻塞性能分析

## 项目目标

在 NSFNET 14 节点拓扑上，研究四种路由策略在经典通信容量和 QKD 密钥容量**双约束**下的阻塞性能：

| 策略 | 说明 |
|------|------|
| `min_hop` | 最小跳数路由 |
| `min_distance` | 最短距离路由 |
| `key_capacity_aware` | 密钥容量感知路由 — 选择密钥容量瓶颈最大的路径 |
| `dual_capacity_aware` | 双资源感知路由 — 选择经典+密钥归一化瓶颈最大的路径 |

每条链路同时具有两类资源，每个业务请求同时具有两类需求。只有当候选路径上所有链路**同时满足**经典带宽和密钥速率需求时，业务才被接受，否则阻塞。

## 与旧项目的关系

本工程是**全新独立项目**，不从属于以下任一旧工程：

- `E:\王雨婷个人文件夹\01：仿真代码合集\QKD_Network`
- `E:\王雨婷个人文件夹\学校统一事务\研一\研究工作\解川-交接材料\代码\KeyConsumption_24node`

旧工程仅作为**只读参考**，用于借鉴：

1. NSFNET 拓扑和链路长度数据（`NSFNET14.json` / `topology1`）
2. BB84 密钥率（SKR）计算函数（`SKR_BB84_finite.py`）
3. 物理层参数设置

本工程**不修改、不删除、不覆盖**旧工程中的任何文件。

## 快速开始

### 环境要求

- Python ≥ 3.10
- 依赖：`networkx`, `numpy`, `pandas`, `matplotlib`

```bash
pip install -r requirements.txt
```

### 默认运行

```bash
cd qkd_dual_capacity_routing
python run.py
```

默认配置：

- QKD 容量模式：`abstract`（指数衰减模型）
- 经典容量模式：`constant`（400 Gb/s）
- 负载范围：20, 40, 60, 80, 100, 120, 140, 160 Erlang
- 请求数：5000 / 负载
- K 最短路径：K=5

运行后在 `results/` 下生成：

- `simulation_results.csv` — 数值结果
- `blocking_rate_vs_load.png` — 总阻塞率
- `key_blocking_rate_vs_load.png` — 密钥阻塞率
- `classical_blocking_rate_vs_load.png` — 经典容量阻塞率
- `avg_path_length_vs_load.png` — 平均路径长度
- `classical_utilization_vs_load.png` — 经典容量利用率
- `key_utilization_vs_load.png` — 密钥容量利用率

## QKD 密钥容量模式

### `abstract`（默认，零依赖）

指数衰减模型：

```python
K = K0 * exp(-alpha * length_km)   # kb/s
```

默认参数：`K0 = 1000 kb/s`, `alpha = 0.02 / km`

⚠️ **注意**：`abstract` 模式仅用于调试和快速验证。正式课程论文结果应切换到 `actual_skr` 模式。

### `actual_skr`（推荐用于正式结果）

通过 adapter 动态加载旧工程中的 BB84 密钥率计算函数（`BB84_SKR_infinite`）。

```bash
python run.py \
    --qkd-capacity-mode actual_skr \
    --old-project-root "../01：仿真代码合集/QKD_Network"
```

要求 `old-project-root` 指向包含 `src/qkd/SKR_BB84_finite.py` 的 `QKD_Network` 项目根目录。

SKR adapter 的工作原理：

1. 运行时通过 `importlib` 动态加载 `SKR_BB84_finite.py`
2. 调用 `BB84_SKR_infinite(distance_m, noise_power_w=0.0)`
3. 返回的 `skr_per_sec`（bps）除以 1000 转换为 kb/s
4. 假设无共传经典信道（`noise_power = 0`）

如果旧工程路径不正确或文件缺失，会抛出 `RuntimeError` 并给出清晰提示——**不会静默降级**为抽象模型。

## 经典容量模式

### `constant`（默认）

```bash
python run.py --classical-capacity-mode constant --constant-classical-cap 400
```

每条边使用相同的固定容量（默认 400 Gb/s）。

### `csv`

从 CSV 文件读取每条边的容量：

```bash
python run.py \
    --classical-capacity-mode csv \
    --classical-capacity-csv data/classical_capacity.csv
```

CSV 格式：`u,v,classical_capacity_gbps`（详见 `data/README.md`）。

### `gnpy_csv`

读取 GNPy / GN-model 工具预先计算的链路质量结果：

```bash
# 直接容量
python run.py \
    --classical-capacity-mode gnpy_csv \
    --gnpy-result-csv data/gnpy_capacity.csv

# GSNR → Shannon 容量映射
python run.py \
    --classical-capacity-mode gnpy_csv \
    --gnpy-result-csv data/gnpy_gsnr.csv \
    --gnpy-bandwidth-ghz 75 \
    --gnpy-osnr-margin-db 3
```

支持两种 CSV 格式（自动检测）：

- **直接容量**：`u,v,classical_capacity_gbps`
- **GSNR**：`u,v,gsnr_db,bandwidth_ghz`（GSNR 通过 Shannon-like 公式映射为容量）

Shannon-like 容量映射：

```python
gsnr_linear = 10 ** (gsnr_db / 10)
margin_linear = 10 ** (margin_db / 10)
capacity_gbps = bandwidth_hz * log2(1 + gsnr_linear / margin_linear) / 1e9
```

> **说明**：该公式是**课程级抽象容量映射**，便于把 GNPy 的传输质量（QoT）结果转化为链路容量，不等同于商用设备净速率。

### `gnpy_optional`

```bash
python run.py --classical-capacity-mode gnpy_optional --gnpy-result-csv data/gnpy.csv
```

与 `gnpy_csv` 相同，但如果 CSV 文件不存在，会**自动回退**到 constant 模式并给出提示。适用于可选 GNPy 集成的场景。

## 配置旧工程路径

```bash
python run.py \
    --qkd-capacity-mode actual_skr \
    --old-project-root "E:\王雨婷个人文件夹\01：仿真代码合集\QKD_Network"
```

## 自定义仿真参数

```bash
python run.py \
    --load-start 10 --load-end 200 --load-step 10 \
    --num-requests 10000 \
    --k-paths 3 \
    --seed 42 \
    --strategies min_hop min_distance dual_capacity_aware
```

完整参数列表：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--qkd-capacity-mode` | QKD 容量模式 | `abstract` |
| `--classical-capacity-mode` | 经典容量模式 | `constant` |
| `--load-start` | 起始负载 (Erlang) | 20 |
| `--load-end` | 结束负载 (Erlang) | 160 |
| `--load-step` | 负载步长 | 20 |
| `--num-requests` | 每负载请求数 | 5000 |
| `--mean-holding-time` | 平均保持时间 | 1.0 |
| `--k-paths` | K 最短路径数 | 5 |
| `--seed` | 随机种子 | 2026 |
| `--strategies` | 路由策略列表 | 全部 4 种 |
| `--output-dir` | 输出目录 | `results` |

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `results/simulation_results.csv` | 每个（负载, 策略）的详细指标 |
| `results/blocking_rate_vs_load.png` | 总阻塞率 vs 负载 |
| `results/key_blocking_rate_vs_load.png` | 密钥容量阻塞率 vs 负载 |
| `results/classical_blocking_rate_vs_load.png` | 经典容量阻塞率 vs 负载 |
| `results/avg_path_length_vs_load.png` | 平均路径长度 vs 负载 |
| `results/classical_utilization_vs_load.png` | 经典容量利用率 vs 负载 |
| `results/key_utilization_vs_load.png` | 密钥容量利用率 vs 负载 |

CSV 列名：

```text
load, strategy, qkd_mode, classical_mode,
num_requests, num_accepted, num_blocked,
blocking_rate, classical_blocking_rate, key_blocking_rate,
joint_blocking_rate, topology_blocking_rate,
avg_hops, avg_path_length_km,
avg_classical_utilization, avg_key_utilization,
max_classical_utilization, max_key_utilization
```

## 模型假设

1. **网络拓扑**：NSFNET 14 节点，21 条无向链路
2. **经典业务速率**：Low={10, 40} Gb/s, Medium={40, 100} Gb/s, High={100, 400} Gb/s
3. **密钥需求**：Low=1 kb/s, Medium=5 kb/s, High=10 kb/s
4. **安全等级分布**：Low 50%, Medium 30%, High 20%
5. **业务到达**：泊松过程（到达间隔指数分布）
6. **业务保持时间**：指数分布，均值 1.0 时间单位
7. **链路约束**：经典容量 + QKD 密钥容量双约束
8. **abstract QKD 模型**：指数衰减 `K = K0 * exp(-α * L_km)` kb/s
9. **actual_skr 模型**：零噪声假设下的 BB84 无限密钥长度 SKR
10. **GSNR→容量映射**：Shannon-like 抽象公式（非商用净速率）

## 阻塞原因分类

每个被阻塞的请求归入以下四类之一：

| 类别 | 含义 |
|------|------|
| `topology_blocking` | 源-宿之间无候选路径（拓扑不连通） |
| `classical_blocking` | 存在密钥容量充足的路径，但经典容量均不足 |
| `key_blocking` | 存在经典容量充足的路径，但密钥容量均不足 |
| `joint_blocking` | 候选路径中经典和密钥容量均无法单独满足 |

判断方法：对 K 条候选路径逐一检查经典可行性（`classical_ok`）和密钥可行性（`key_ok`），汇总后分类。

## 项目结构

```text
qkd_dual_capacity_routing/
  README.md
  requirements.txt
  run.py                          # CLI 主入口
  qkd_routing/
    __init__.py
    config.py                     # SimulationConfig + 默认参数
    topology.py                   # NSFNET 拓扑 + KSP 预计算
    traffic.py                    # 业务请求生成 (Poisson)
    routing.py                    # 四种路由策略
    resources.py                  # 双资源管理 (EdgeResources)
    simulation.py                 # 离散事件仿真引擎
    metrics.py                    # 结果聚合 + CSV 输出
    plotting.py                   # Matplotlib 绘图
    skr_adapter.py                # QKD 密钥容量 (abstract / actual_skr)
    gnpy_adapter.py               # 经典容量 (constant / csv / gnpy_csv)
    utils.py                      # 共用工具函数
  data/
    README.md                     # CSV 数据格式说明
  results/                        # 仿真输出 (CSV + PNG)
```
