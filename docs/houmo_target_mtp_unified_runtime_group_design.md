# HouMo Target 与 MTP 统一 Runtime Group 详细设计

## 1. 文档信息

- **状态**：设计稿
- **适用仓库**：`llama.cpp`
- **基线提交**：`e5bd53f676341f26366a9f4944655f55da019666` 及其后续 Qwen3.5/Gemma4 MTP 实现
- **核心目标**：将相同 HMM group、相同 replica 的 Target 与 MTP 组成一个统一的 `RuntimeGroup`，保证加载、初始化、sequence 路由、内存映射和生命周期严格成对。

---

## 2. 背景

提交 `e5bd53f676341f26366a9f4944655f55da019666` 引入了多 HMM group 支持。当前 Target 路径会扫描：

```text
prefill.hmm      + decode.hmm      -> group 0
prefill_1.hmm    + decode_1.hmm    -> group 1
prefill_2.hmm    + decode_2.hmm    -> group 2
```

每个 group 创建一个独立 `HouMoLLModel::Impl`，并通过 global `seq_id` 取模路由到对应 Impl。

后续 MTP 实现虽然也引入了 `mtp_group_ids_` 和 `impls_mtp_`，但当前仍限制：

- 只允许 MTP group 0；
- 只创建一个 MTP Impl；
- MTP Impl 固定绑定 `impls_.front()`；
- MTP 初始化只允许 `seq_max == 1`；
- Target 和 MTP 使用两套平行容器和两套初始化状态。

这无法支持以下目标形态：

```text
Runtime 0: Target(group 0, replica 0) + MTP(group 0, replica 0)
Runtime 1: Target(group 0, replica 1) + MTP(group 0, replica 1)
Runtime 2: Target(group 1, replica 0) + MTP(group 1, replica 0)
```

因此需要把 Target/MTP 从“两个独立 Impl 列表”重构为“同一 Runtime Group 内的一对运行实例”。

---

## 3. 术语

### 3.1 HMM Group

由 HMM tensor 文件名后缀确定的一组模型能力。一个完整 group 可包含：

```text
Target:
  prefill[_N].hmm
  decode[_N].hmm 或 decoder[_N].hmm

MTP:
  prefill_mtp[_N].hmm
  decode_mtp[_N].hmm 或 decoder_mtp[_N].hmm
```

其中无后缀表示 group 0，`_1` 表示 group 1。

### 3.2 Replica

同一 HMM group 的独立运行时副本。副本之间可以共享只读权重资源，但必须独立持有：

- TCIM Module；
- KV/TCIM cache；
- valid length；
- recurrent state；
- checkpoint pool；
- pending verify/draft 状态；
- 输入输出 tensor 状态。

### 3.3 Runtime Group

一条完整 speculative inference lane，包含同一 `group_id`、同一 `replica_id` 的 Target 和 MTP：

```text
RuntimeGroup(group_id, replica_id)
  ├── target Impl
  ├── mtp Impl（可选）
  ├── target memory adapter
  └── mtp memory adapter
```

### 3.4 Runtime Index

`RuntimeGroup` 在 `runtime_groups_` 中的索引。sequence 路由必须以 runtime index 为单位，而不是以 HMM group id 为单位。

---

## 4. 设计目标

### 4.1 功能目标

1. Target 和 MTP 按 `(group_id, replica_id)` 严格成对。
2. 支持多个 Target/MTP HMM group。
3. 支持同一个 HMM group 创建多个独立 replica。
4. 同一 global `seq_id` 在 Target context 与 MTP context 中路由到同一个 runtime group。
5. 保持当前 `ctx_tgt` 和 `ctx_dft` 两个 llama context 的上层 speculative 编排不变。
6. 保持不同 group 使用不同 context length 的能力。
7. 保持 MTP rollback 仅回退逻辑 valid length 的语义。
8. 加载失败、初始化失败和析构均不存在泄漏、悬空引用或 double free。

### 4.2 非目标

本阶段不负责：

- 将不同 runtime 的 TCIM `Run()` 自动并行化；
- 动态迁移活跃 sequence 到其他 runtime；
- 根据实时负载动态切换 replica；
- 修改 llama.cpp 通用 speculative decoding 算法；
- 将 Target 与 MTP 合并为单个 `llama_context`；
- 物理清除 rejected MTP cache tail。

---

## 5. 当前架构与问题

## 5.1 当前数据结构

`HouMoLLModel` 当前维护两套平行结构：

```cpp
std::vector<Impl *> impls_;
std::vector<Impl *> impls_mtp_;
std::vector<int> group_ids_;
std::vector<int> mtp_group_ids_;
std::vector<int> local_seq_max_;
std::vector<int> local_seq_max_mtp_;
std::vector<std::unique_ptr<houmo_memory_i>> memories_;
std::vector<std::unique_ptr<houmo_memory_i>> memories_mtp_;
```

问题包括：

1. Target/MTP 对应关系由 vector 下标隐式表达，无法保证一致。
2. `group_index` 同时承担 HMM group index 和 runtime instance index 两种语义。
3. 当前 MTP 固定使用 `impls_.front()` 作为 target。
4. 多 replica 后同一 group id 会重复，group id 不再能唯一标识 runtime。
5. 裸指针需要手工清理，部分加载失败路径复杂。
6. Target/MTP context 分开初始化，缺少共享且稳定的 route topology。

## 5.2 当前加载限制

当前 MTP 加载逻辑：

```cpp
mtp_group_ids_ = collect_hmm_groups(ml, true);
if (mtp_group_ids_.size() != 1 || mtp_group_ids_.front() != 0) {
    // only group 0
}
Impl *impl = create_impl(ml.arch_name, true, impls_.front());
```

存在两个关键错误：

- 多 group 时所有 MTP 不可能绑定对应 Target；
- 多 replica 时无法区分 Target replica。

## 5.3 当前初始化限制

当前 MTP 初始化要求：

```cpp
seq_max == 1 && impls_mtp_.size() == 1
```

这与 server 多 slot、多 runtime 的目标不兼容。

## 5.4 当前上层关系

server 对同一个 `llama_model` 创建两个 context：

```text
ctx_tgt: LLAMA_CONTEXT_TYPE_DEFAULT
ctx_dft: LLAMA_CONTEXT_TYPE_MTP
```

二者共享同一个 `model.hm_llmodel`，但分别调用：

```cpp
hm_llmodel->houmo_init(..., is_mtp=false)
hm_llmodel->houmo_init(..., is_mtp=true)
```

当前 `HouMoLLModel::houmo_init()` 还通过单一 `memory_` 成员要求 Target 与
MTP 传入相同的 memory 指针：

```cpp
if (memory_ == nullptr) {
    memory_ = memory;
} else if (memory_ != memory) {
    return false;
}
```

但 `llama_context::houmo_init()` 会为 `ctx_tgt` 和 `ctx_dft` 分别创建一个
`houmo_memory_i`。因此该检查与双 context 的实际所有权模型冲突，统一
Runtime Group 重构时必须删除单一 `memory_` 指针约束，改为分别记录
Target/MTP root memory，并仅要求二者具有相同的 runtime topology 和
global/local sequence 映射。

因此设计必须允许：

- Target runtime 和 MTP runtime 分别初始化；
- 二者共享完全相同的 runtime group 拓扑；
- 两次初始化使用不同的 `houmo_memory_i` 根对象；
- 同一 global `seq_id` 的 route 结果不可发生变化。

---

## 6. 总体方案

将“模型描述”和“运行时状态”分成两层：

```text
HmmGroupDescriptor
  描述 GGUF 中有哪些 Target/MTP HMM 文件、需要多少 replica

RuntimeGroup
  表示一个具体运行 lane，拥有独立 Target/MTP Impl 和 context-specific memory
```

统一调用路径：

```text
global seq_id
  -> route_seq(seq_id)
  -> runtime_groups_[runtime_index]
       ├── target impl + target local seq
       └── mtp impl    + mtp local seq
```

路由只计算一次逻辑，不再接受 `is_mtp` 来选择不同列表。

---

## 7. HMM 文件分组规则

## 7.1 Target 文件名

支持：

```text
prefill.hmm
prefill_1.hmm
prefill_2.hmm

decode.hmm / decoder.hmm
decode_1.hmm / decoder_1.hmm
decode_2.hmm / decoder_2.hmm
```

## 7.2 MTP 文件名

支持：

```text
prefill_mtp.hmm
prefill_mtp_1.hmm
prefill_mtp_2.hmm

decode_mtp.hmm / decoder_mtp.hmm
decode_mtp_1.hmm / decoder_mtp_1.hmm
decode_mtp_2.hmm / decoder_mtp_2.hmm
```

兼容旧模型：

```text
assistant.hmm / assistant.hmms
```

旧 assistant 文件只映射到 group 0。若存在多 Target group，同时仅存在旧 assistant 文件，则加载 MTP 模式时应报错，不允许隐式复用到所有 group。

## 7.3 完整性规则

当模型不包含任何 MTP tensor 时：

- 允许仅 Target group；
- 每个 `RuntimeGroup::mtp == nullptr`；
- `has_mtp_modules()` 返回 false。

当模型包含任意 MTP tensor 时，默认采用严格模式：

```text
Target group set == MTP group set
```

例如下面情况应加载失败：

```text
Target groups = {0, 1}
MTP groups    = {0}
```

错误信息应明确指出缺失 group：

```text
MTP HMM groups do not match target groups: target={0,1}, mtp={0}, missing_mtp={1}
```

严格一一对应可以避免 speculative slot 被路由到无 MTP 的 runtime。

---

## 8. Replica 配置

## 8.1 配置语义

每个 HMM group 可以配置独立 replica 数：

```text
group 0 -> 2 replicas
group 1 -> 1 replica
```

展开后：

```text
runtime 0 = group 0 replica 0
runtime 1 = group 0 replica 1
runtime 2 = group 1 replica 0
```

## 8.2 配置来源

推荐优先级：

1. 显式运行参数；
2. GGUF metadata；
3. 默认值 1。

建议 metadata：

```text
houmo.hmm.replica_count              = 1
houmo.hmm.group.0.replica_count      = 2
houmo.hmm.group.1.replica_count      = 1
```

组级配置覆盖全局配置。

第一阶段若暂不修改公共参数，可先只实现 metadata；不建议长期使用环境变量作为正式接口。

## 8.3 配置校验

- replica count 必须大于 0；
- 总 runtime 数不得超过 `n_seq_max`；
- 每个 Impl 必须通过 `supports_seq_max(local_seq_max)`；
- 同 group 所有 replica 加载后能力必须一致。

---

## 9. 数据结构设计

## 9.1 HMM Group 描述

在 `src/houmo/houmo-llmodel.h` 增加：

```cpp
struct HmmGroupDescriptor {
    int group_id = -1;
    int replica_count = 1;
    bool has_target_prefill = false;
    bool has_target_decode = false;
    bool has_mtp_prefill = false;
    bool has_mtp_decode = false;
};
```

如果暂时不需要保存 tensor 名，可以只保存能力标志；实际 Impl 继续通过 `load_group_id_` 过滤 GGUF tensor。

## 9.2 Runtime Group

```cpp
struct RuntimeGroup {
    int runtime_index = -1;
    int group_id = -1;
    int replica_id = -1;

    std::unique_ptr<Impl> target;
    std::unique_ptr<Impl> mtp;

    std::unique_ptr<houmo_memory_i> target_memory;
    std::unique_ptr<houmo_memory_i> mtp_memory;

    int target_local_seq_max = 0;
    int mtp_local_seq_max = 0;
    bool target_initialized = false;
    bool mtp_initialized = false;
};
```

容器：

```cpp
std::vector<RuntimeGroup> runtime_groups_;
```

删除：

```cpp
impls_;
impls_mtp_;
group_ids_;
mtp_group_ids_;
local_seq_max_;
local_seq_max_mtp_;
memories_;
memories_mtp_;
```

## 9.3 Route

```cpp
struct SeqRoute {
    int runtime_index = -1;
    int group_id = -1;
    int replica_id = -1;
    int local_seq_id = -1;
    RuntimeGroup *runtime = nullptr;
};
```

提供 const/non-const 两个版本：

```cpp
bool route_seq(int global_seq_id, SeqRoute &route);
bool route_seq(int global_seq_id, ConstSeqRoute &route) const;
```

或使用内部模板避免重复。

路由不再接收 `is_mtp`。调用方根据 context 类型选择 `route.runtime->target` 或 `route.runtime->mtp`。

---

## 10. 路由设计

## 10.1 基础算法

保持当前静态、sticky、可逆路由：

```cpp
runtime_index = global_seq_id % runtime_count;
local_seq_id  = global_seq_id / runtime_count;
```

逆映射：

```cpp
global_seq_id = local_seq_id * runtime_count + runtime_index;
```

示例：

| global seq | runtime | group | replica | local seq |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 | 0 |
| 2 | 2 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 1 |
| 4 | 1 | 0 | 1 | 1 |
| 5 | 2 | 1 | 0 | 1 |

## 10.2 Target/MTP 路由一致性

对同一个 global `seq_id`，Target 和 MTP 必须使用同一 `route_seq()`：

```text
route_seq(4)
  -> runtime 1
  -> group 0 replica 1
  -> local seq 1

Target: runtime.target, local seq 1
MTP:    runtime.mtp,    local seq 1
```

禁止：

- Target 与 MTP 使用不同 runtime 数；
- Target 按 group 路由、MTP 固定 group 0；
- MTP 使用独立取模公式；
- 活跃 sequence 运行期间修改 runtime group 顺序。

## 10.3 路由拓扑冻结

`runtime_groups_` 必须在 `houmo_load()` 完成后冻结。Target/MTP context 初始化、请求运行和 sequence reset 均不得增删或重排 runtime。

---

## 11. Memory Adapter 设计

当前 `GroupMemory` 实际映射的是 runtime index，不是 HMM group。重命名为：

```cpp
class RuntimeMemory final : public houmo_memory_i
```

成员：

```cpp
RuntimeMemory(
    houmo_memory_i *root,
    int runtime_index,
    int runtime_count);
```

映射：

```cpp
llama_seq_id global(llama_seq_id local) const {
    return local * runtime_count_ + runtime_index_;
}
```

Target context 和 MTP context 各自有独立 root memory，因此每个 RuntimeGroup 保存两个 adapter：

```text
target_memory -> ctx_tgt root memory
mtp_memory    -> ctx_dft root memory
```

两者映射公式相同，但底层 memory 对象不同。

`HouMoLLModel` 可分别保存非 owning root 指针用于初始化一致性校验：

```cpp
houmo_memory_i *target_root_memory_ = nullptr;
houmo_memory_i *mtp_root_memory_ = nullptr;
```

二者不要求地址相同；只要求各角色只初始化一次、`seq_max` 一致且
`runtime_groups_` 顺序不变。当前单一 `memory_` 成员应删除。

此设计保证：

- Target 与 MTP logical position 分别维护；
- 相同 global seq 映射到相同 runtime/local seq；
- `common_context_seq_rm(ctx_dft, seq_id, ...)` 能正确回退对应 MTP replica；
- rejected MTP tail 只通过 MTP memory valid length 回退，无需清物理 cache。

---

## 12. 加载流程设计

## 12.1 发现 group

将当前 `collect_hmm_groups(loader, mtp)` 重构为：

```cpp
std::vector<HmmGroupDescriptor>
discover_hmm_groups(const llama_model_loader &loader);
```

扫描一次即可同时建立 Target/MTP 集合，减少两次独立扫描导致的信息不一致。

流程：

1. 扫描所有 GGUF tensor 名；
2. 按 group id 设置 Target/MTP 标志；
3. 校验 Target prefill/decode 配对；
4. 校验 MTP prefill/decode 配对；
5. 检测是否启用 MTP；
6. 若启用，校验 Target/MTP group set 严格一致；
7. 读取每组 replica count；
8. 按 group id 升序输出 descriptor。

## 12.2 创建 Runtime Group

伪代码：

```cpp
for (const auto &desc : descriptors) {
    for (int replica_id = 0; replica_id < desc.replica_count; ++replica_id) {
        RuntimeGroup runtime;
        runtime.runtime_index = runtime_groups_.size();
        runtime.group_id = desc.group_id;
        runtime.replica_id = replica_id;

        runtime.target.reset(create_impl(arch, false));
        configure_shared_resources(*runtime.target);
        runtime.target->set_load_group_id(desc.group_id);
        if (!runtime.target->houmo_load(ml, device_ids)) {
            return false;
        }

        if (desc.has_mtp_prefill || desc.has_mtp_decode) {
            runtime.mtp.reset(
                create_impl(arch, true, runtime.target.get()));
            configure_shared_resources(*runtime.mtp);
            runtime.mtp->set_load_group_id(desc.group_id);
            if (!runtime.mtp->houmo_load(ml, device_ids)) {
                return false;
            }
        }

        validate_runtime_pair(runtime);
        runtime_groups_.push_back(std::move(runtime));
    }
}
```

## 12.3 Target/MTP 绑定

对于 Gemma4：

```cpp
new HouMoGemma4MTP(
    static_cast<HouMoGemma4 *>(runtime.target.get()))
```

必须绑定同一个 RuntimeGroup 的 Target。

对于 Qwen3.5，当前 `HoumoQwen35MTP` 构造不依赖 target 指针，但仍必须放入同一 RuntimeGroup，以保证 sequence/cache 路由一致。未来若 Qwen MTP 需要共享 Target tensor，不需要再次调整架构。

## 12.4 共享资源

不同 runtime 可以共享：

- `tcim::Module::WeightManager`；
- `HoumoEmbeddingLayer`；
- 只读模型 metadata。

不得共享：

- `Impl` 对象；
- prefill/decode Module 运行状态；
- KV/state cache；
- valid length map；
- checkpoint pool；
- pending verify；
- MTP AR hidden state。

## 12.5 异常安全

使用 `std::unique_ptr` 后，加载失败只需：

```cpp
runtime_groups_.clear();
```

不再维护 `fail_load` 中的手工 `delete` 循环。

加载采用临时容器，全部成功后提交更安全：

```cpp
std::vector<RuntimeGroup> new_groups;
// load all
runtime_groups_ = std::move(new_groups);
```

这样失败不会留下半加载状态。

---

## 13. 初始化流程设计

## 13.1 Target context 初始化

当 `is_mtp == false`：

1. 检查 `target_initialized_ == false`；
2. 检查 runtime group 非空；
3. 检查 `seq_max >= runtime_count`；
4. 按 runtime index 分配 local capacity；
5. 对每个 Target Impl 调用 `supports_seq_max()`；
6. 为每组创建 Target `RuntimeMemory`；
7. 初始化 Target Impl；
8. 保存 `target_local_seq_max`；
9. 设置 `target_seq_max_` 和 `target_initialized_`。

local capacity：

```cpp
local_seq_max = (seq_max + runtime_count - 1 - runtime_index)
                / runtime_count;
```

## 13.2 MTP context 初始化

当 `is_mtp == true`：

1. 检查 `mtp_initialized_ == false`；
2. 检查每个 RuntimeGroup 都有 MTP Impl；
3. 检查 `seq_max` 与 Target 拓扑兼容；
4. 使用与 Target 完全相同的 runtime count 和 local capacity 公式；
5. 对每个 MTP Impl 调用 `supports_seq_max()`；
6. 创建 MTP `RuntimeMemory`；
7. 初始化 MTP Impl；
8. 保存 `mtp_local_seq_max`；
9. 设置 `mtp_seq_max_` 和 `mtp_initialized_`。

## 13.3 两个 context 的 seq_max 一致性

MTP speculative 模式下建议强制：

```text
target_seq_max == mtp_seq_max
```

Target 先初始化、MTP 后初始化时：

```cpp
if (target_initialized_ && seq_max != target_seq_max_) {
    return false;
}
```

MTP 先初始化的反向情况同理。

这样确保相同 global seq 在两个 context 中都合法。

## 13.4 初始化事务性

当前 `Impl::houmo_init()` 返回 `void`，无法可靠传播中途初始化失败。长期建议改成：

```cpp
virtual bool houmo_init(...)
```

本次若控制改动范围，可保留异常传播，但必须保证：

- 每个 Impl 初始化失败时抛异常；
- `HouMoLLModel::houmo_init()` 只在全部成功后设置 initialized flag；
- 已初始化的前序 runtime 能在 model/context 销毁时安全释放。

---

## 14. 推理调用调整

## 14.1 Target Prefill

```cpp
SeqRoute route;
route_seq(seq_id, route);
return route.runtime->target->houmo_prefill(
    tokens, route.local_seq_id, logits);
```

## 14.2 Target Decode

按 `runtime_index` 分桶：

```cpp
std::map<int, std::vector<size_t>> batches_by_runtime;
```

禁止按 `group_id` 分桶，因为同一 HMM group 的不同 replica 拥有不同状态。

每桶调用：

```cpp
runtime_groups_[runtime_index].target->houmo_decode(...)
```

## 14.3 Target Verify/Accept

`houmo_verify()` 和 `houmo_verify_accept()` 使用同一 route，并调用 `runtime.target`。

Qwen3.5 checkpoint/state commit 保持 Impl 内部逻辑不变。

## 14.4 MTP Decode

```cpp
SeqRoute route;
route_seq(seq_id, route);
if (!route.runtime->mtp) {
    return -1;
}
return route.runtime->mtp->houmo_decode_mtp(
    h_in,
    tokens,
    route.local_seq_id,
    logits_out,
    h_out,
    start_pos);
```

## 14.5 Context length 查询

```cpp
uint32_t n_context_length(int seq_id) const {
    SeqRoute route;
    return route_seq(seq_id, route)
        ? route.runtime->target->n_context_length()
        : 0;
}
```

slot context length始终由 Target group 决定。MTP group 加载后应校验其 context 能覆盖 Target speculative 使用范围，但不作为 server slot 长度来源。

## 14.6 Batch 能力查询

`n_decode_batch()`、`n_prefill_batch()` 遍历所有 `runtime.target` 取最小值，保持当前全局保守语义。

如未来希望不同 slot 使用不同 batch 能力，可增加按 seq 查询 API，本阶段不做。

---

## 15. Target Hidden 与 MTP 配对

上层 `common_speculative_impl_draft_mtp` 仍保留：

```text
ctx_tgt 输出 pre-norm hidden
ctx_dft 消费 (hidden, token) pair
```

只要两个 context 对相同 `seq_id` 使用同一 runtime topology，就能保证：

```text
Target hidden from group G replica R
  -> MTP group G replica R
```

### 15.1 `set_pre_norm_hidden_dst()` 风险

当前实现向所有 Target Impl 广播同一个输出地址：

```cpp
for (auto *impl : impls_) {
    impl->set_pre_norm_hidden_dst(dst, n_rows);
}
```

单 sequence 调用时通常只有一个 Impl 执行，因此可工作。但多个 runtime 同 batch 时，可能出现多个 Impl 写同一 buffer 起始位置。

统一 Runtime Group 后建议改为调用级路由：

1. `llama_context::houmo_decode()` 在拆分 batch 时计算每个 runtime 的输出行偏移；
2. 仅对实际执行的 `runtime.target` 设置对应 dst；
3. Impl 完成后清空 dst，避免悬空地址。

若本阶段不改，至少增加约束：HouMo HMM hidden 输出批次不得跨 runtime，并添加断言。

---

## 16. MTP Cache 与 Rollback 语义

该重构不改变 MTP cache 语义。

对于 speculative draft：

- 每个 `RuntimeGroup::mtp` 独立持有 cache；
- draft 期间 valid length 连续增长；
- verify 接受 `A` 个 draft 后，下轮逻辑位置只增长 `A + 1`；
- rejected tail 可以保留在物理 TCIM cache；
- `common_context_seq_rm(ctx_dft, seq_id, accepted_end, -1)` 通过 MTP `RuntimeMemory` 回退逻辑长度；
- 下一次 MTP 调用必须继续满足：

```text
past_seq_length = start_pos - 1
```

不同 replica 之间不得共享：

- `mtp_past_seq_len`；
- `mtp_cache_end`；
- decode valid length；
- MTP hidden AR state。

---

## 17. 生命周期设计

## 17.1 所有权

```text
llama_model
  -> shared_ptr<HouMoLLModel>
      -> vector<RuntimeGroup>
          -> unique_ptr<Target Impl>
          -> unique_ptr<MTP Impl>
```

两个 `llama_context` 共享同一个 `HouMoLLModel`，分别初始化 Target/MTP memory。

## 17.2 析构

`RuntimeGroup` 使用 `unique_ptr` 后无需手工 delete。成员析构顺序应保证 MTP 先于 Target 析构，尤其 Gemma4 MTP 持有非 owning `target_` 指针。

C++ 成员按声明逆序析构。建议声明顺序：

```cpp
std::unique_ptr<Impl> target;
std::unique_ptr<Impl> mtp;
```

逆序会先析构 `mtp`，再析构 `target`，符合依赖关系。

## 17.3 Context 销毁问题

当前 Target/MTP runtime 初始化状态保存在 `llama_model` 级对象中，而不是 `llama_context`。这意味着同一个 model 不能重复创建第二套同类型 context。

本设计保持该限制，不扩大范围。长期建议将 RuntimeGroup 的可变 runtime 状态下沉到 `llama_context`，model 只保留共享权重和 group descriptor。

---

## 18. 并发与线程安全

多 replica 首先提供状态隔离，不自动意味着并发执行。

当前 `HouMoLLModel::houmo_decode()` 对 runtime 桶顺序执行。若后续并行，必须先确认：

1. 不同 `tcim::Module` 可以并发 `Run()`；
2. 共享 WeightManager 支持并发；
3. embedding layer 是只读或线程安全；
4. 日志/performance counter 无数据竞争；
5. hidden 输出 buffer 已按 runtime 分片；
6. LoRA 状态更新与运行互斥；
7. abort callback 生命周期安全。

本阶段维持顺序执行，避免引入线程问题。

---

## 19. 资源与容量

每增加一个 RuntimeGroup replica，以下资源近似线性增加：

- Target Module runtime；
- Target KV/recurrent state；
- Target checkpoint pool；
- MTP Module runtime；
- MTP cache；
- host/device IO buffer。

只读权重通过 WeightManager 共享，不代表 runtime cache 不增长。

初始化日志建议增加：

```text
runtime=1 group=0 replica=1 target_ctx=262144 mtp_ctx=4096 local_seq_max=1
```

并汇总：

```text
runtime_group_count=3 target_impls=3 mtp_impls=3
```

---

## 20. 错误处理

以下情况必须明确失败：

1. Target group 缺少 prefill/decode 配对；
2. MTP group 缺少 prefill/decode 配对；
3. Target/MTP group set 不一致；
4. replica count <= 0；
5. `seq_max < runtime_group_count`；
6. Target 与 MTP context 的 `seq_max` 不一致；
7. MTP context 初始化时某 RuntimeGroup 没有 MTP Impl；
8. 同 group replica 的 context/batch 能力不一致；
9. Gemma4 MTP target 类型转换失败；
10. route 得到的 local seq 超出 Target 或 MTP capacity；
11. MTP 调用发生在未初始化的 MTP runtime；
12. Target/MTP runtime topology 被重复加载或重排。

错误日志统一包含：

```text
runtime_index, group_id, replica_id, global_seq_id, local_seq_id
```

---

## 21. 兼容性设计

## 21.1 单 group、无 MTP

```text
prefill.hmm + decode.hmm
```

行为与当前一致：一个 RuntimeGroup，仅 Target。

## 21.2 单 group、带 MTP

```text
prefill.hmm + decode.hmm
prefill_mtp.hmm + decode_mtp.hmm
```

默认一个 RuntimeGroup，保持当前 Qwen3.5 MTP 场景。

## 21.3 多 context length

```text
group 0: 32K Target + 32K-compatible MTP
group 1: 256K Target + 256K-compatible MTP
```

server slot 仍通过 `llama_n_ctx_seq_id(ctx_tgt, slot.id)` 获取路由 Target 的 context length。

## 21.4 同 context 多 replica

不需要复制 HMM tensor，只通过 replica count 创建多个独立 runtime pair。

## 21.5 旧 assistant HMM

仅单 group 0 时兼容；多 Target group 时拒绝模糊映射。

---

## 22. 文件与符号级修改清单

## 22.1 `src/houmo/houmo-llmodel.h`

修改：

- 增加 `HmmGroupDescriptor`；
- 增加 `RuntimeGroup`；
- 重构 `SeqRoute`；
- `route_seq()` 删除 `is_mtp` 参数；
- 用 `runtime_groups_` 替换 Target/MTP 平行 vectors；
- 保留 Target/MTP initialized 和 seq_max 状态；
- 可增加诊断 API：runtime count、group id、replica id。

## 22.2 `src/houmo/houmo-llmodel.cpp`

修改：

- `collect_hmm_groups()` -> `discover_hmm_groups()`；
- `GroupMemory` -> `RuntimeMemory`；
- `create_impl()` 保留，但 MTP 必须接收本 RuntimeGroup Target；
- `HouMoLLModel::~HouMoLLModel()` 删除手工 delete；
- 重写 `houmo_load()`；
- 重写 `houmo_init()`；
- 重写 `route_seq()`；
- Target prefill/decode/verify/accept 改走 `runtime.target`；
- MTP decode 改走 `runtime.mtp`；
- context/batch/embedding/LoRA 查询遍历 runtime target；
- `has_mtp_modules()` 检查所有 RuntimeGroup 的 MTP 完整性。

## 22.3 `src/llama-context.cpp`

主要保持接口不变，增加：

- Target/MTP context seq_max 一致性错误传播；
- MTP context 的 `cparams.n_ctx` 不应错误覆盖 server Target slot 长度；
- hidden 输出跨 runtime 的路由/断言；
- 初始化日志输出 runtime topology。

## 22.4 `src/houmo/houmo-gemma4mtp.{h,cpp}`

确认：

- 每个 MTP Impl 的 `target_` 指向同 RuntimeGroup Target；
- target cache tensor 绑定不跨 replica；
- MTP 析构先于 Target；
- 加载/初始化日志打印 group/replica。

## 22.5 `src/houmo/houmo-qwen35mtp.{h,cpp}`

确认：

- 每个 replica 的 cache tracker 独立；
- rollback 只依赖该 replica 的 RuntimeMemory valid length；
- 日志增加 runtime/group/replica；
- `past_seq_length = start_pos - 1` 关系保持不变。

## 22.6 `tools/server/server-context.cpp`

原则上无需改变 speculative 调用链。建议增加：

- 创建 `ctx_dft` 后校验 `llama_n_seq_max(ctx_tgt) == llama_n_seq_max(ctx_dft)`；
- slot 初始化时打印 route group/replica（需要新增 HouMo 内部查询 API）；
- 保持 `slot.n_ctx` 使用 Target `llama_n_ctx_seq_id()`。

## 22.7 `common/speculative.cpp`

算法无需修改。建议增加调试校验：

- HouMo HMM 场景下 Target/MTP route topology 一致；
- `ctx_tgt` 与 `ctx_dft` 的 `n_seq_max` 一致；
- MTP begin/draft/accept 日志带 global seq id。

---

## 23. 测试设计

## 23.1 单元测试：group 发现

| 用例 | Target | MTP | 期望 |
|---|---|---|---|
| 单 Target group | `{0}` | `{}` | 成功，无 MTP |
| 单完整 group | `{0}` | `{0}` | 成功 |
| 多完整 group | `{0,1}` | `{0,1}` | 成功 |
| Target 缺 decode | prefill 0 | 无 decode 0 | 失败 |
| MTP 缺 decode | mtp prefill 0 | 无 mtp decode 0 | 失败 |
| group set 不一致 | `{0,1}` | `{0}` | 失败 |
| assistant 兼容 | `{0}` | assistant | 成功 |
| assistant 多组歧义 | `{0,1}` | assistant | 失败 |

## 23.2 单元测试：runtime 展开

配置：

```text
group 0 replicas=2
group 1 replicas=1
```

验证顺序：

```text
runtime 0 -> g0r0
runtime 1 -> g0r1
runtime 2 -> g1r0
```

每个 runtime 应拥有不同 Target/MTP Impl 地址。

## 23.3 单元测试：路由

`runtime_count=3, seq_max=6`：

| global seq | runtime | local seq |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 0 | 1 |
| 4 | 1 | 1 |
| 5 | 2 | 1 |

Target 与 MTP route 必须完全相同。

## 23.4 初始化测试

- Target 先、MTP 后；
- MTP 先、Target 后；
- 两边相同 seq_max 成功；
- 两边 seq_max 不同失败；
- `seq_max < runtime_count` 失败；
- 某 MTP Impl 不支持 local capacity 时失败；
- 重复初始化同一种 context 失败。

## 23.5 Target-only 回归

覆盖：

- prefill；
- decode；
- verify；
- verify accept；
- LoRA；
- embedding/ASR 模型；
- 不同 context length slot。

确保不启用 MTP 时行为不变。

## 23.6 MTP 功能测试

每个 runtime 至少测试：

1. prompt prefill；
2. 首轮 MTP decode；
3. 4/4 全接受；
4. 2/4 部分接受；
5. 0/4 全拒绝；
6. rollback 后下一轮 valid length；
7. EOS/最大 token 截断；
8. sequence reset 后重新请求。

关键断言：

```text
MTP past_seq_length = start_pos - 1
next_start_pos = previous_start_pos + accepted_drafts + 1
```

## 23.7 多 replica 隔离测试

两个并发 slot 路由到同 group 不同 replica：

```text
slot 0 -> g0r0
slot 1 -> g0r1
```

验证：

- cache 地址不同；
- valid length 独立；
- checkpoint pool 独立；
- 一个 slot rollback 不影响另一个 slot；
- sequence reset 不影响其他 replica；
- 输出与单 slot baseline 一致。

## 23.8 多 group + 多 replica 测试

```text
group 0: 32K, replicas=2
group 1: 256K, replicas=1
parallel=3
```

验证：

- slot 0/1 为 32K，slot 2 为 256K；
- 每个 slot 的 Target/MTP group 和 replica 相同；
- 长 prompt 只发送到可容纳的 slot；
- speculative decoding 正常；
- server context shift 边界正确。

## 23.9 Gemma4 专项

- 每个 MTP target 指针指向本 runtime Target；
- shared cache tensor 不跨 replica；
- Target 先销毁不会发生；
- 多 runtime 初始化后 cache binding 数量正确。

## 23.10 稳定性测试

- 连续请求 1000 次；
- 多 slot 混合长短 prompt；
- 高频全拒绝；
- prompt cache save/load；
- slot erase/copy/shift；
- 服务退出资源释放；
- ASAN/UBSAN（Host 部分）；
- device/host 内存增长监控。

---

## 24. 日志设计

加载日志：

```text
loaded runtime=0 group=0 replica=0 target=yes mtp=yes
loaded runtime=1 group=0 replica=1 target=yes mtp=yes
loaded runtime=2 group=1 replica=0 target=yes mtp=yes
```

初始化日志：

```text
initialized target runtime=1 group=0 replica=1 local_seq_max=1
initialized mtp runtime=1 group=0 replica=1 local_seq_max=1
```

路由 trace：

```text
route global_seq=1 -> runtime=1 group=0 replica=1 local_seq=0 role=mtp
```

MTP valid length：

```text
[MTP-DECODE] runtime=1 group=0 replica=1 global_seq=1 local_seq=0 start_pos=41 past_seq_len=40
```

日志级别建议：加载/初始化为 INFO，逐 token 路由为 TRACE/DEBUG。

---

## 25. 实施阶段

### 阶段一：统一数据模型

- 引入 `RuntimeGroup` 和 `RuntimeMemory`；
- Target/MTP 使用同一 group 容器；
- 保持默认每 group 一个 replica；
- 删除 MTP group 0 限制；
- 保持顺序执行。

验收：多 Target/MTP group 可以加载、初始化和路由。

### 阶段二：同 group 多 replica

- 增加 replica count 配置；
- 展开 RuntimeGroup；
- 增加 group/replica 日志；
- 完成多 slot 隔离测试。

验收：同一组 HMM 无需复制 tensor，即可创建多个 Target/MTP pair。

### 阶段三：hidden 输出与批处理加固

- hidden dst 按 runtime 分片；
- 多 runtime batch 显式分桶；
- 增加跨 runtime batch 测试。

验收：多 slot batch 下 hidden/token 配对无覆盖。

### 阶段四：并发执行（可选）

- 验证 TCIM/WeightManager 线程安全；
- runtime 桶并行运行；
- 性能与资源调优。

---

## 26. 验收标准

设计实现完成后必须满足：

1. GGUF 中每个 Target HMM group 都与同 id MTP group 配对；
2. 每个 replica 创建独立 Target/MTP Impl；
3. MTP Impl 绑定同 RuntimeGroup Target；
4. Target/MTP context 对同 global seq 路由一致；
5. 多 context length 行为不退化；
6. 同 context length 支持多 replica；
7. MTP partial/zero acceptance 后 valid length 正确回退；
8. 一个 replica 的 cache/state 操作不影响其他 replica；
9. 单 group、单 slot、无 MTP 等旧场景回归通过；
10. 加载失败和服务退出无泄漏、double free 或悬空 target 指针。

---

## 27. 最终推荐形态

```text
HouMoLLModel
  ├── shared WeightManager
  ├── shared EmbeddingLayer
  ├── descriptors
  │    ├── group 0, replicas=2
  │    └── group 1, replicas=1
  └── runtime_groups
       ├── runtime 0: group 0 replica 0
       │    ├── Target Impl
       │    └── MTP Impl -> Target Impl
       ├── runtime 1: group 0 replica 1
       │    ├── Target Impl
       │    └── MTP Impl -> Target Impl
       └── runtime 2: group 1 replica 0
            ├── Target Impl
            └── MTP Impl -> Target Impl
```

该结构将 HMM 模型能力、运行时副本和 sequence 路由清晰分离，同时保留当前 llama.cpp 双 context speculative 编排，是支持“不同 context length + 同 context 多 Impl + MTP 成对运行”的最小且可持续架构。
