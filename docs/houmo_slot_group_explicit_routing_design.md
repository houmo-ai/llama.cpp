# HouMo Slot 与 Context Group 显式映射设计

> 日期：2026-08-08  
> 状态：设计阶段  
> 适用范围：HouMo backend、`llama-server`、`llama-threaded-server`  
> 目标：支持不同 HMM context group 绑定不等数量的 server slot

## 1. 背景

HouMo GGUF 可以包含多个 target HMM group。每个 group 对应一套独立的 HouMo runtime/TCIM Module 和本地 sequence 空间。

llama-server 中：

- `n_parallel` 决定 server slot 数量；
- slot ID 为 `[0, n_parallel)`；
- server 将 slot ID 直接作为 llama.cpp 的 global `seq_id`；
- HouMo backend 再把 global `seq_id` 映射为某个 HMM group 的 local `seq_id`。

当前映射完全由取余和整除计算：

```cpp
group_index = global_seq_id % group_count;
local_seq_id = global_seq_id / group_count;
```

相应地，`GroupMemory` 使用逆公式将 local `seq_id` 转回 global `seq_id`：

```cpp
global_seq_id = local_seq_id * group_count + group_index;
```

这种映射只支持近似均匀、交错分配，不能表达不同 group 拥有不同 slot 容量的情况。

例如：

```text
n_parallel = 10
GGUF target groups = [0, 2]

期望：
  group 0 绑定 2 个 slot
  group 2 绑定 8 个 slot
```

期望的实际映射是：

```text
slot 0 -> group 0, local seq 0
slot 1 -> group 0, local seq 1
slot 2 -> group 2, local seq 0
slot 3 -> group 2, local seq 1
...
slot 9 -> group 2, local seq 7
```

当前取余算法无法表达该关系。

## 2. 设计目标

本设计需要满足：

1. 每个 server slot 可以显式绑定到一个 target HMM group；
2. 不同 group 可以绑定不同数量的 slot；
3. group ID 不要求连续，例如可以是 `[0, 2]`；
4. group 内部 local `seq_id` 必须连续且从 0 开始；
5. global-to-local 与 local-to-global 映射必须互为逆映射；
6. 未提供配置时保持当前取余行为，兼容已有模型和启动参数；
7. 映射属于 context 实例配置，不影响 TCIM WeightManager 共享；
8. 同一 GGUF 创建的不同 context 可以使用不同映射；
9. slot save/restore、KV 操作、checkpoint、MTP 等现有功能不能产生错路由；
10. 配置错误必须在 context 初始化阶段失败，不能等到第一次请求才暴露。

## 3. 非目标

本阶段不包括：

- 在请求执行过程中动态迁移 slot 到另一个 group；
- 根据负载自动调整 group slot 数量；
- 跨 group 共享 KV cache；
- 让同一个 slot 同时绑定多个 group；
- 修改 TCIM WeightManager 的共享规则；
- 通过进程级环境变量配置不同线程实例；
- 修改 server slot 的调度和选择算法。

## 4. 术语

### 4.1 Global sequence ID

llama.cpp context 所看到的 sequence ID。在 server 中等同于 slot ID：

```text
global_seq_id == server_slot.id
```

取值范围：

```text
[0, n_parallel)
```

### 4.2 Group ID

GGUF 中 HMM group 的实际编号，例如：

```text
prefill.hmm / decoder.hmm       -> group ID 0
prefill_2.hmm / decoder_2.hmm   -> group ID 2
```

Group ID 不一定连续，不能直接作为 `impls_` 数组下标。

### 4.3 Group index

group 在当前 `HouMoLLModel::impls_` 中的数组下标。

例如：

```text
group_ids_ = [0, 2]

group ID 0 -> group index 0
group ID 2 -> group index 1
```

### 4.4 Local sequence ID

特定 HMM group 内部使用的 sequence ID，范围为：

```text
[0, local_seq_max[group_index])
```

## 5. 配置方案

### 5.1 配置归属

Slot/group 映射属于 `llama_context` 配置，而不是模型权重配置。

原因：

- 相同权重可以创建多个不同 `n_parallel` 的 context；
- 不同 context 可以采用不同 slot 分配；
- 映射不改变 HMM 权重内容；
- 映射不应加入 WeightManager pool key；
- 多线程 Router 中环境变量无法按实例隔离。

推荐传递链路：

```text
模型 preset / CLI
    ↓
common_params
    ↓
llama_context_params
    ↓
llama_context::houmo_init()
    ↓
HouMoLLModel::houmo_init()
```

### 5.2 主要配置：完整 slot map

建议新增参数：

```text
--houmo-slot-group-map <group-id-list>
```

配置值是按 slot ID 排列的 group ID 列表：

```bash
--houmo-slot-group-map 0,0,2,2,2,2,2,2,2,2
```

含义：

```text
数组下标 = global slot/seq ID
数组值   = HMM group ID
```

在 preset 中可以写成：

```ini
houmo-slot-group-map = 0,0,2,2,2,2,2,2,2,2
```

该格式可以表达任意静态映射，包括非连续分配：

```ini
houmo-slot-group-map = 0,2,2,0,2,2,2,2,2,2
```

### 5.3 可选简写：group capacity

可以额外支持：

```text
--houmo-group-slots 0:2,2:8
```

表示按配置顺序进行连续分配：

```text
group 0 获得前 2 个 slot
group 2 获得后 8 个 slot
```

内部必须将其展开为完整 slot map：

```text
[0,0,2,2,2,2,2,2,2,2]
```

完整 map 应作为唯一内部表示。

首版建议只实现 `--houmo-slot-group-map`，避免两个配置来源之间的冲突规则。容量简写可以后续添加。

### 5.4 默认行为

未提供显式配置时，自动生成与当前取余算法完全相同的映射：

```cpp
for (int slot = 0; slot < n_parallel; ++slot) {
    group_id_map[slot] = group_ids_[slot % group_count];
}
```

这样可以移除运行期的取余计算，同时保持已有行为。

## 6. 参数数据结构

### 6.1 common_params

建议增加：

```cpp
struct common_params {
    // ...
    std::vector<int32_t> houmo_slot_group_map;
};
```

CLI/preset 解析阶段将逗号分隔字符串解析为整数数组。

### 6.2 llama_context_params

该配置需要在创建 context 前传入 HouMo backend。

建议在 fork 内增加：

```cpp
struct llama_context_params {
    // ...
    const int32_t * houmo_slot_group_map;
    size_t          houmo_slot_group_map_count;
};
```

`common_context_params_to_llama()` 负责设置指针和数量。

必须确保数组生命周期覆盖 `llama_init_from_model()` 调用。`llama_context` 构造时应复制数组，不能长期保存来自 `common_params` 的裸指针。

如果不希望扩展 public `llama_context_params` ABI，可以引入 HouMo fork 专用 context 创建 API。但从当前代码结构和维护成本看，在该 fork 内扩展 context params 更直接。

### 6.3 llama_context 内部

建议复制为：

```cpp
std::vector<int32_t> houmo_slot_group_map;
```

然后调用：

```cpp
hm_llmodel->houmo_init(
    memory,
    cparams.n_seq_max,
    abort_callback,
    abort_callback_data,
    is_mtp,
    houmo_slot_group_map);
```

## 7. HouMo 路由数据结构

### 7.1 当前结构的问题

当前只保存：

```cpp
std::vector<int> group_ids_;
std::vector<int> local_seq_max_;
```

`route_seq()` 每次根据取余计算路由，没有保存显式关系。

`GroupMemory` 也根据公式反推 global ID。这两处都依赖均匀交错布局。

### 7.2 推荐结构

在 `HouMoLLModel` 中增加：

```cpp
struct SeqRoute {
    int group_index  = -1;
    int group_id     = -1;
    int local_seq_id = -1;
    Impl * impl      = nullptr;
};

std::vector<SeqRoute> target_routes_;
std::vector<SeqRoute> mtp_routes_;

std::vector<std::vector<llama_seq_id>> target_local_to_global_;
std::vector<std::vector<llama_seq_id>> mtp_local_to_global_;
```

其中：

- `target_routes_[global_seq_id]` 完成 global-to-local 查询；
- `target_local_to_global_[group_index][local_seq_id]` 完成 local-to-global 查询。

### 7.3 示例

输入：

```text
group_ids_ = [0, 2]
n_parallel = 10
slot_group_map = [0,0,2,2,2,2,2,2,2,2]
```

构造结果：

```text
target_routes_[0] = {index=0, id=0, local=0, impl=impls_[0]}
target_routes_[1] = {index=0, id=0, local=1, impl=impls_[0]}
target_routes_[2] = {index=1, id=2, local=0, impl=impls_[1]}
target_routes_[3] = {index=1, id=2, local=1, impl=impls_[1]}
...
target_routes_[9] = {index=1, id=2, local=7, impl=impls_[1]}

target_local_to_global_[0] = [0,1]
target_local_to_global_[1] = [2,3,4,5,6,7,8,9]

local_seq_max_ = [2,8]
```

## 8. 路由表构建算法

建议在 `HouMoLLModel::houmo_init()` 中调用独立 helper：

```cpp
bool build_seq_routes(
    int seq_max,
    const std::vector<int> & group_ids,
    const std::vector<Impl *> & impls,
    const std::vector<int32_t> & configured_slot_group_map,
    std::vector<SeqRoute> & routes,
    std::vector<std::vector<llama_seq_id>> & local_to_global,
    std::vector<int> & local_seq_max,
    std::string & error);
```

算法：

1. 建立 `group_id -> group_index` 查找表；
2. 如果用户没有配置，生成默认取余 map；
3. 校验 map 长度等于 `seq_max`；
4. 初始化每个 group 的 `local_to_global` 数组；
5. 按 global slot ID 从 0 到 `seq_max - 1` 遍历；
6. 查找配置的 group ID 对应的 group index；
7. local ID 等于该 group 当前已分配 slot 数量；
8. 写入 global-to-local route；
9. 将 global ID 追加到 local-to-global；
10. 遍历结束后得到每个 group 的 `local_seq_max`；
11. 调用每个 Impl 的 `supports_seq_max()`；
12. 所有校验通过后再初始化任何 group runtime。

伪代码：

```cpp
for (int global = 0; global < seq_max; ++global) {
    const int group_id = slot_group_map[global];
    auto it = group_id_to_index.find(group_id);
    if (it == group_id_to_index.end()) {
        return error("unknown group id");
    }

    const int group_index = it->second;
    const int local = local_to_global[group_index].size();

    routes[global] = {
        group_index,
        group_id,
        local,
        impls[group_index],
    };
    local_to_global[group_index].push_back(global);
}
```

## 9. route_seq() 修改

修改后不再执行取余：

```cpp
bool HouMoLLModel::route_seq(
        int global_seq_id,
        bool is_mtp,
        SeqRoute & route) const {
    const auto & routes = is_mtp ? mtp_routes_ : target_routes_;
    if (global_seq_id < 0 || global_seq_id >= (int) routes.size()) {
        return false;
    }
    route = routes[global_seq_id];
    return route.impl != nullptr;
}
```

查询复杂度仍然为 O(1)，并且避免每次 decode 进行除法和取余。

## 10. GroupMemory 修改

### 10.1 当前问题

当前构造参数：

```cpp
GroupMemory(memory, group_index, group_count)
```

逆映射：

```cpp
global = local * group_count + group_index;
```

显式路由后该公式不再成立。

### 10.2 新结构

建议改成：

```cpp
class GroupMemory final : public houmo_memory_i {
public:
    GroupMemory(
        houmo_memory_i * memory,
        std::vector<llama_seq_id> local_to_global)
        : memory_(memory),
          local_to_global_(std::move(local_to_global)) {}

private:
    llama_seq_id global(llama_seq_id local) const {
        if (local < 0 || local >= (llama_seq_id) local_to_global_.size()) {
            throw std::out_of_range("invalid local HouMo sequence id");
        }
        return local_to_global_[local];
    }

    houmo_memory_i * memory_;
    std::vector<llama_seq_id> local_to_global_;
};
```

所有以下操作会自动使用正确映射：

- `seq_rm()`；
- `seq_cp()`；
- `seq_keep()`；
- `seq_add()`；
- `seq_div()`；
- `seq_pos_min()`；
- `seq_pos_max()`；
- `state_write()`；
- `state_read()`。

## 11. Group 初始化容量

当前容量使用均匀公式：

```cpp
local_seq_max = (seq_max + count - 1 - i) / count;
```

修改后直接使用显式映射统计结果：

```cpp
local_seq_max[i] = local_to_global[i].size();
```

对于示例：

```text
group 0 -> houmo_init(seq_max=2)
group 2 -> houmo_init(seq_max=8)
```

在创建任何 `GroupMemory` 或调用 `Impl::houmo_init()` 之前，必须先完成所有 group 的 `supports_seq_max()` 校验，确保初始化过程具有事务性，避免部分 group 已分配资源后才发现配置错误。

## 12. 零 Slot Group

配置可能没有给某个已加载 group 分配 slot，例如：

```text
GGUF groups = [0, 2]
map = [2,2,2,2]
```

首版建议禁止这种情况，要求每个 target group 至少绑定一个 slot。

原因：

- 当前 Impl 初始化通常假定 `seq_max > 0`；
- 即使 group 不使用，HMM Module 已经在 model load 阶段加载；
- 跳过 runtime 初始化会增加 state、LoRA、MTP 等路径的条件分支；
- 配置错误更容易被及时发现。

后续如果确实需要闲置 group，可以单独设计 lazy runtime initialization。

## 13. MTP 路由

当前 MTP 约束：

```text
mtp group 数量 = 1
mtp group ID = 0
seq_max = 1
```

首版设计建议：

- 显式 slot/group map 只应用于 target context；
- MTP context 保持固定 `{global 0 -> group 0, local 0}`；
- 如果对 MTP context 提供了非空 target map，忽略并打印 debug 日志，或者直接拒绝；
- 更推荐直接拒绝，避免用户误以为配置生效。

未来 MTP 支持多 group 时，再复用同一套 route builder。

## 14. State Save/Restore

HouMo sequence state header 当前保存：

```text
global seq_id
group ID
local seq ID
```

需要明确两种恢复语义。

### 14.1 恢复到相同 slot

如果配置没有变化：

- global seq ID 相同；
- group ID 相同；
- local seq ID 相同；
- 可以正常恢复。

### 14.2 恢复到不同 slot/group

现有代码注释允许 source 和 destination slot 位于不同 physical group，并由模型 payload 校验 tensor 兼容性。

恢复时不应要求 header 中的 group/local 与当前 destination route 完全相同。应当：

1. 使用调用参数 `seq_id` 查询当前 destination route；
2. header group/local 仅表示 state 来源；
3. destination Impl 负责校验 tensor name、shape、stride 和 dtype；
4. 成功后更新 global memory position。

因此显式路由不应改变 state 文件格式版本，除非后续要保存完整映射快照。

### 14.3 配置变化

重启后映射发生变化时，slot state 可能恢复到不同 group。只要两个 group state schema 兼容，可以继续支持；否则恢复应失败并保持 destination state 不变。

需要新增测试覆盖这一场景。

## 15. Server Slot 稳定性

server 初始化时创建：

```text
slot.id = [0, n_parallel)
```

推理、KV、checkpoint 和 sampler 路径都使用 `slot.id` 作为 `seq_id`，因此当前 slot ID 足够稳定，可以直接作为映射表索引。

需要注意 `get_slot_by_id()` 当前允许对越界 ID 取模。该行为发生在 server API 层，进入 HouMo backend 的最终 slot ID 仍然位于合法范围内。

本设计不修改 server slot 调度。请求未指定 slot 时，server 仍然选择任意可用 slot，因此请求最终使用哪个 group 取决于被分配到哪个 slot。

如果业务需要请求明确选择 group，应通过固定 `id_slot`、slot pool 分类或后续新增 group-aware scheduler 实现，不应让 HouMo backend 在请求开始后移动 sequence。

## 16. 配置校验

初始化阶段必须校验：

1. map 元素数量等于 `n_parallel/n_seq_max`；
2. 每个 group ID 都存在于当前 GGUF target group 集合；
3. 每个 target group 至少出现一次；
4. 每个 group 的统计数量通过 `supports_seq_max()`；
5. group ID 为非负整数；
6. 配置中不存在空元素；
7. 配置没有整数溢出；
8. MTP context 没有错误使用 target map；
9. embedding-only 模型的 group 配置符合其单 group 约束；
10. 显式 map 与未来的 capacity 简写不能同时出现。

错误信息需要包含模型、slot、group 和允许值，例如：

```text
invalid HouMo slot group map: slot 4 references group 1,
but model groups are [0,2]
```

```text
invalid HouMo slot group map: expected 10 entries for n_parallel=10,
got 8
```

```text
HouMo group 0 cannot initialize seq_max=2: <backend reason>
```

## 17. 日志与可观测性

context 初始化时打印最终展开后的路由：

```text
HouMo slot routing: mode=explicit, n_seq_max=10, groups=2
HouMo slot routing: group_id=0 group_index=0 local_seq_max=2 slots=[0,1]
HouMo slot routing: group_id=2 group_index=1 local_seq_max=8 slots=[2,3,4,5,6,7,8,9]
```

默认兼容模式：

```text
HouMo slot routing: mode=round-robin-default
```

debug 级别可以按 slot 打印：

```text
HouMo slot route: global=2 -> group_id=2 group_index=1 local=0
```

不建议在每次 token decode 时打印路由，避免影响性能和日志可读性。

可以在 `/props?model=...` 中增加只读信息：

```json
{
  "houmo_slot_routing": {
    "mode": "explicit",
    "slot_group_map": [0,0,2,2,2,2,2,2,2,2],
    "groups": [
      {"id": 0, "slots": [0,1]},
      {"id": 2, "slots": [2,3,4,5,6,7,8,9]}
    ]
  }
}
```

该接口不是首版必需项，但有助于线上确认实际生效配置。

## 18. 与多线程 Router 的关系

多线程 Router 中每个模型 preset 创建独立 `server_context`，因此每个实例可以拥有不同的 slot/group map：

```text
实例 A：同一 GGUF，n_parallel=10，map=0,0,2,2,2,2,2,2,2,2
实例 B：同一 GGUF，n_parallel=4， map=0,2,2,2
```

两个实例仍然可以共享 TCIM WeightManager，因为映射只影响 context-local runtime、KV 和 slot，不影响权重 identity。

该配置不能通过环境变量实现 per-instance 隔离。多线程中的环境变量属于整个进程，必须通过 preset/`common_params` 显式传递。

## 19. API 与配置示例

### 19.1 CLI

```bash
./bin/llama-server \
    -m model.gguf \
    -np 10 \
    --houmo-slot-group-map 0,0,2,2,2,2,2,2,2,2
```

### 19.2 threaded router preset

示意：

```ini
[qwen3-context-layout-a]
model = /models/qwen3.gguf
n-parallel = 10
houmo-slot-group-map = 0,0,2,2,2,2,2,2,2,2

[qwen3-context-layout-b]
model = /models/qwen3.gguf
n-parallel = 4
houmo-slot-group-map = 0,2,2,2
```

两个 preset 使用同一个模型文件，可以共享 WeightManager，但各自使用独立的路由表、Module runtime 和 KV。

## 20. 兼容与迁移

### 20.1 无配置

继续生成当前 round-robin 映射，行为不变。

### 20.2 有配置

启用 explicit 模式。配置不合法时启动失败。

### 20.3 State 文件

首版不修改 state header 格式，不需要提升 `HOUMO_STATE_VERSION`。

### 20.4 权重共享

不修改 WeightManager key，也不影响 device 权重共享。

### 20.5 原多进程 Router

原 Router 的 child preset 同样能够携带该参数，因此单模型、多进程 Router 和多线程 Router 可以共用同一配置语义。

## 21. 实现步骤

建议按以下顺序实现。

### 阶段一：参数链路

1. 在 `common_params` 增加 vector 字段；
2. 增加 CLI/preset 参数解析；
3. 在 usage 中说明数量必须等于 `n_parallel`；
4. 将配置复制到 context params；
5. 添加纯参数解析单元测试。

### 阶段二：路由表

1. 实现独立 `build_seq_routes()`；
2. 无配置时生成当前 round-robin map；
3. 有配置时解析 group ID；
4. 生成 global-to-local 和 local-to-global；
5. 生成 `local_seq_max_`；
6. 替换 `route_seq()` 取余逻辑。

### 阶段三：Memory 反向映射

1. 修改 `GroupMemory` 构造参数；
2. 删除 `group_count_` 逆公式；
3. 使用 local-to-global vector；
4. 增加 local ID 边界检查。

### 阶段四：初始化与校验

1. 在任何 runtime 初始化前构建和校验全部路由；
2. 对每个 group 调用 `supports_seq_max()`；
3. 使用统计容量调用 `Impl::houmo_init()`；
4. 输出最终路由日志。

### 阶段五：测试与文档

1. 添加单元测试；
2. 添加 server 集成测试；
3. 添加 state save/restore 测试；
4. 添加 threaded router 多实例测试；
5. 更新 Router 和 HouMo 使用文档。

## 22. 测试设计

### 22.1 路由表单元测试

#### 默认双 group

```text
groups=[0,2], np=6, no config
expected map=[0,2,0,2,0,2]
```

验证：

```text
slot 0 -> group 0 local 0
slot 1 -> group 2 local 0
slot 2 -> group 0 local 1
slot 3 -> group 2 local 1
```

#### 非均匀显式分配

```text
groups=[0,2], np=10
map=[0,0,2,2,2,2,2,2,2,2]
```

验证：

```text
local_seq_max=[2,8]
group0 local_to_global=[0,1]
group2 local_to_global=[2,3,4,5,6,7,8,9]
```

#### 非连续显式分配

```text
map=[0,2,2,0,2]
```

验证：

```text
group0 local_to_global=[0,3]
group2 local_to_global=[1,2,4]
```

#### 非法 group

```text
groups=[0,2]
map=[0,1,2]
```

必须失败。

#### 长度不匹配

```text
np=10
map entries=8
```

必须失败。

#### group 无 slot

```text
groups=[0,2]
map=[2,2,2]
```

首版必须失败。

### 22.2 GroupMemory 测试

给定：

```text
local_to_global=[2,3,7]
```

验证所有 memory 操作将：

```text
local 0 -> global 2
local 1 -> global 3
local 2 -> global 7
```

特别覆盖：

- `seq_rm`；
- `seq_add`；
- `seq_pos_min/max`；
- state write/read。

### 22.3 Decode 分组测试

一次 batch 同时包含：

```text
slots=[0,2,3,1]
```

验证 `houmo_decode()` 按 group 拆分后传入：

```text
group 0 local IDs=[0,1]
group 2 local IDs=[0,1]
```

并验证 logits 按原 batch 顺序写回。

### 22.4 Server 集成测试

- `np=10` 启动成功；
- 固定请求到 slot 0/1，确认走 group 0；
- 固定请求到 slot 2~9，确认走 group 2；
- 并发占满全部 slot；
- 保存、恢复、清除每个 slot；
- context shift/checkpoint 不串 group。

### 22.5 多线程 Router 测试

同一 GGUF 加载两个 preset：

```text
A: np=10, map=0,0,2,2,2,2,2,2,2,2
B: np=4,  map=0,2,2,2
```

验证：

- 两个实例各自路由正确；
- WeightManager 日志显示复用；
- 一个实例 unload 不影响另一个实例路由；
- 两个实例 state/KV 不串扰。

## 23. 风险分析

### 23.1 只修改 route_seq

如果只修改 global-to-local，而不修改 `GroupMemory`，KV 和 state 操作会使用错误的 global slot。这是最高风险点，必须成对修改。

### 23.2 配置在加载后改变

映射决定 KV ownership，context 运行期间不能热更新。任何变更都必须卸载并重建 context。

### 23.3 Slot 调度与业务预期

server 默认选择任意空闲 slot。即使配置正确，也不能保证普通请求一定进入某个 group。如果 group 代表不同能力或 SLA，需要额外的 group-aware slot scheduler。

### 23.4 State 跨 group 兼容

映射变化可能使 state 恢复到不同 group。必须依赖严格 tensor schema 校验，失败时保持 destination state 原子性。

### 23.5 Public ABI

扩展 `llama_context_params` 会改变结构体 ABI。该仓库是定制 fork 时风险可控，但仍需要同步所有初始化点和 bindings。

## 24. 最终建议

首版采用以下最小完整方案：

1. 新增 `--houmo-slot-group-map`；
2. 配置放在模型 preset/CLI，不引入独立全局配置文件；
3. 未配置时生成当前取余等价 map；
4. context 初始化时一次性构建双向路由表；
5. `route_seq()` 改为 O(1) 查表；
6. `GroupMemory` 使用显式 local-to-global 数组；
7. group runtime 容量按实际 slot 数量初始化；
8. 首版要求每个已加载 target group 至少一个 slot；
9. MTP 暂时保持单 group 固定映射；
10. context 运行期间禁止热更新映射。

该方案能够正确支持：

```text
np=10
group 0 -> 2 slots
group 2 -> 8 slots
```

同时保持现有默认行为、WeightManager 共享和 server slot 语义不变。

