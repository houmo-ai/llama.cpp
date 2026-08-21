# llama-server 请求优先级与 NPU 抢占设计

## 1. 目标与结论

需求：请求携带整数优先级，`0` 为最高优先级，数值越大优先级越低；高优先级请求到达后，低优先级请求暂停当前 `llama_decode`，释放 NPU 计算资源；高优先级请求完成后，低优先级请求从暂停点继续。本设计限定为 HMM 模型、单 batch decode；`n_parallel > 1` 时请求是交替执行，不做多 batch 合并。

结论：当前 `llama-server` **不支持该语义**。

现有代码具备部分可复用能力，但只能实现“下一轮调度优先”，不能实现 decode 中途抢占：

* `tools/server/server-queue.{h,cpp}` 只有 FIFO 队列和 `post(..., front=true)`，没有优先级字段、排序或抢占逻辑。
* `server_context::process_single_task()` 在有空闲 slot 时启动任务，无空闲 slot 时放入 deferred 队列；运行中的 slot 不会被新任务驱逐。
* `update_slots()` 将多个 slot 合成 batch，并在循环中调用 `llama_decode()`。一次 `llama_decode()` 返回前，server 主推理线程不会处理新任务。
* `llama_context_params.abort_callback` 可以中止 `llama_decode()`，但 API 明确标注目前只对 CPU 执行有效，不能作为 NPU 抢占方案；且中止后的 batch/KV 一致性需额外处理。
* 代码已有 context checkpoint、slot save/restore、KV cache sequence 操作，可作为暂停/恢复的基础，但当前 checkpoint 主要用于上下文窗口管理，并非通用调度快照。

因此，推荐实现“**跨 slot 协作式抢占**”：不保存或迁移 slot 状态，只在调度时停止把低优先级 slot 放入下一次 HMM decode；slot 的 KV/recurrent/sampler 状态原地保留。该方案不依赖 NPU 驱动强制中断。

## 2. HMM 场景下的简化假设

HMM 场景下按优先级分层调度：调度器找出当前活动 slot 的最高优先级，所有同优先级 slot 可以并发进入 decode；低优先级 slot 在 step 边界让出。无需保存跨 slot 快照、驱逐 slot 或做 batch 内 token 重排。被暂停 slot 的 KV/recurrent/sampler 状态保持在原 slot 中，恢复时直接继续。服务不要求使用 `--no-cont-batching`。

如果 HMM 后端的 `houmo_decode` 一次调用内部仍包含多个不可拆分的 NPU 运算，则抢占粒度仍然是一次 `houmo_decode` 返回；应将其拆成最小可恢复调用，或接受该调用时长作为抢占延迟上界。

## 3. 多线程与操作系统线程挂起分析

不建议通过操作系统 API 直接挂起 `llama-server` 线程来实现抢占，原因如下：

1. 挂起 CPU 线程不会自动撤销已经提交给 NPU 的 command queue、kernel 或 DMA。NPU 仍可能继续占用资源，高优先级线程未必能获得设备执行权。
2. 如果暂停发生在 `llama_decode` 内部，线程可能持有 ggml、KV cache、后端 runtime 或 server mutex；挂起后高优先级线程访问同一 context 可能死锁或破坏状态。
3. POSIX `SIGSTOP`/`SIGSUSPEND` 一类机制通常按进程或线程组生效，难以只暂停一个请求；`pthread` 没有可移植的安全挂起 API。Linux `tgkill`、`ptrace` 等做法不可移植，也无法提供 C++ 对象不变量保证。
4. 多线程 server 若多个线程共享同一个 `llama_context`/NPU context，线程级挂起不能形成请求级隔离；若每请求独立 context，则挂起仍无法释放已分配的 KV/设备内存，也不能保证 NPU runtime 支持并发抢占。
5. 强制杀线程会丢失 sampler、HMM recurrent state 和 KV 状态，恢复只能重新推理，无法满足“从暂停点继续”。

操作系统调度只能作为辅助措施：将推理线程设置为较低 CPU nice/实时优先级，或把高低优先级请求放到不同进程；它不能替代 NPU runtime 的抢占和 llama 状态快照。

正确做法是由 server 自己在 HMM decode step 边界协作式暂停：设置原子抢占标志，当前 step 正常返回后将低优先级 slot 标记为不可调度、不再把它加入下一次 `llama_decode`，并由同一 inference scheduler 线程运行高优先级 slot。这里不做状态保存；slot 的上下文状态原地保留。

## 4. 非目标与约束

本设计不承诺在单个 NPU kernel 执行期间获得硬实时抢占；抢占延迟至少为当前微批（`n_ubatch`）及已提交 NPU 工作完成时间。也不改变模型权重共享、KV cache 总容量或 HTTP 流式协议的基本语义。

若 HouMo/NPU 后端内部把一次 decode 拆成多个 decode 调用，应在这些调用之间增加同样的调度检查；若后端只暴露不可拆分的大调用，只能降低 batch 大小来缩短最长停顿。

## 5. 对外协议

在 completion、infill、embedding、rerank 等请求 JSON 中增加可选字段：

```json
{"priority": 0}
```

建议约束：整数范围 `[0, 2^31-1]`，缺省值为 `100`（保持现有请求近似公平；也可选择 `0` 以保持兼容，需在发布说明中明确）。优先级只影响排队和运行中 slot 的选择，不改变同一请求内部 token 顺序。

响应和 `/metrics` 增加：`priority`、排队时长、抢占次数、累计暂停时长、最近一次抢占原因。取消请求必须能取消排队中、已暂停和运行中的任务。

## 6. 推荐架构

### 6.1 数据结构

在 `server_task` 和 `server_slot` 增加：

```cpp
int32_t priority = 100;
uint64_t enqueue_seq = 0;       // 同优先级 FIFO
uint32_t preempt_count = 0;
slot_state state;               // RUNNING / PAUSED 等
```

将 `queue_tasks`、`queue_tasks_deferred` 改为按 `(priority asc, enqueue_seq asc)` 获取最小元素的优先队列（或保留 deque、插入时二分排序）。deferred 队列也必须按优先级选择，不能因旧请求长期占据队头而产生优先级反转。

### 6.2 调度点

在 `update_slots()` 中把一次大 batch 拆成可配置的微批，默认 `preempt_quantum = n_ubatch`，每次微批前后执行：

1. 检查是否有更高优先级的可运行任务。
2. 若有，当前 `llama_decode` 返回后，将低优先级 slot 转为 `PAUSED`，从下一轮 batch/step 的候选集合移除。
3. 立即只选择高优先级 slot 调用 `llama_decode`，从而不再向 NPU 提交低优先级 slot 的工作。
4. 高优先级 slot 结束后，将暂停 slot 恢复为 `RUNNABLE`。

不要在持有 server queue mutex 时调用 NPU 或 `llama_decode`，避免 HTTP 入队线程阻塞。

### 6.3 暂停与恢复

暂停点必须位于成功的 `llama_decode` step 之后。跨 slot 抢占不需要保存或复制状态：slot/task 标识、token 游标、采样器、KV、HMM recurrent state、LoRA/grammar/工具调用状态都继续保留在原 slot/context 中。暂停仅改变调度状态，并将 slot 从活动 decode 候选集中移除。必须确认 llama backend 允许在同一 context 中保留多个 sequence 的 KV，同时只提交其中一个 sequence；若后端会因“长期未使用”自动清理 KV，需要关闭该行为或增加 slot pin 标记。

恢复时重新选择原 slot，直接从其现有 token 游标、sampler、KV 和 HMM recurrent state 继续；不得重新发送已经向客户端发送过的 token。

### 6.4 资源与并发模型

保持单一 inference scheduler 线程作为 NPU 提交者，HTTP 线程只负责解析、入队和响应。调度器每轮先处理控制任务，再按优先级填充 batch。高优先级请求到达后设置原子 `preempt_requested`，调度器在下一个微批边界处理它。

当多个活动 slot 同时运行时，只有优先级低于队首可运行任务的 slot 才可被暂停；同优先级任务采用轮转，防止饥饿。建议增加 `priority_aging_ms`，等待时间达到阈值后临时提升有效优先级。

## 5. 关键代码改动点

1. `tools/server/server-task.h/cpp`：解析和序列化 `task_params.priority`，校验范围。
2. `tools/server/server-queue.h/cpp`：优先级队列、稳定 FIFO tie-break、查询最高等待优先级。
3. `tools/server/server-context.cpp`：
   * HTTP handler 创建 task 时填充 priority；
   * `process_single_task()` 为暂停 slot/空闲 slot 选择策略；
   * `update_slots()` 微批循环、抢占请求检查、暂停/恢复；
   * 取消、slot 释放、错误路径清理快照。
4. `server_slot`（通常在 `server-context.h` 定义）：增加 PAUSED 状态和 snapshot 句柄。
5. `include/llama.h`/后端：仅在验证 NPU 支持后考虑扩展 abort/同步接口；第一版不要假设 `abort_callback` 对 NPU 生效。
6. metrics、README 和 OpenAI-compatible schema：记录字段和限制。

## 6. 抢占流程

```text
HTTP 入队(priority=0)
        |
优先级队列发现运行中的 priority=5
        |
等待当前微批 decode 返回
        |
保存 slot KV + sampler + token 游标，slot=PAUSED
        |
启动 priority=0 请求（可独占 NPU）
        |
高优先级请求完成/释放资源
        |
恢复 PAUSED slot，继续下一微批
```

“高优先级任务执行结束”应定义为任务完成、取消或失败；仅 prompt 完成不算结束，因为生成阶段仍占用 NPU。

## 7. 失败处理与边界条件

* 单个微批太大时，抢占延迟上界较大；提供 `--preempt-quantum` 并在 NPU 后端实测调优。
* KV 快照空间不足：暂停失败时可降级为不抢占并记录指标，或拒绝新高优先级请求；默认不丢失低优先级请求。
* 多 completion/child task 必须作为一个可抢占组保存和恢复，避免只暂停父任务造成状态不一致。
* speculative、grammar、工具调用和多模态输入必须纳入一致性测试；图片处理等非 decode 操作也要在循环中检查抢占标志。
* 同一请求重复抢占时保持单一快照，禁止嵌套快照；恢复后清除标志并递增计数。

## 8. 分阶段实施

### Phase 0：可观测性（低风险）

加入 priority 字段、优先级队列、队列和 slot 指标，但暂不暂停运行中任务；验证协议兼容和排序正确性。

### Phase 1：边界抢占（推荐首版）

限制 `n_ubatch`/`preempt_quantum`，在 decode 微批边界暂停 slot；先使用进程内 KV checkpoint，覆盖 CPU 和已验证的 NPU backend。

### Phase 2：设备状态优化

针对 NPU 实现 pinned-host KV snapshot、异步拷贝和资源复用；测量暂停/恢复开销并决定是否支持多个暂停上下文。

### Phase 3：策略增强

加入 aging、租约/最大连续运行时间、租户配额和优先级反转保护。

## 9. 验证计划

* 单元测试：优先级排序、同优先级 FIFO、取消排队/暂停任务、快照生命周期。
* 集成测试：低优先级长生成期间提交高优先级请求，验证高优先级首 token 延迟、低优先级输出无重复/丢失。
* NPU 压测：1/2/4 个并发 slot，不同微批大小，统计 P50/P95 抢占延迟、吞吐下降、设备内存峰值。
* 故障注入：decode 返回错误、快照不足、NPU 拷贝失败、客户端断开、服务退出恢复。
* 兼容性：未携带 `priority` 的请求行为与现有版本一致；流式和非流式 API 均覆盖。

## 10. 最终建议

先合并 Phase 0，再针对实际 NPU backend 做 Phase 1 原型。不要直接依赖 `abort_callback` 实现 NPU 抢占；必须以“微批边界 + 可恢复 KV/采样器快照”为正确性基础，并把最大抢占延迟、快照内存上限和不支持的 backend 明确写入文档及 API 能力探测结果。
