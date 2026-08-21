# HMM 请求优先级抢占测试记录

## 1. 测试环境

* 模型：`/home/user/dev/models/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf`
* 可执行文件：`build3/bin/llama-server`
* 设备：HoumoNPU LQ50 DUO-48GB，检测到 device 0 和 device 1
* CPU：12th Gen Intel(R) Core(TM) i5-12600K
* 服务参数：

```bash
build3/bin/llama-server \
  -m /home/user/dev/models/HiModel_xh2_qwen3_0.6b_256_32k_b1_1chip_2cores_v1.2.0_20260422.gguf \
  --parallel 2 \
  --metrics \
  --host 127.0.0.1 \
  --port 18080
```

服务日志确认：

* `n_slots = 2`
* 每个 slot `n_ctx = 32768`
* 模型被识别并正常加载
* `server is listening on http://127.0.0.1:18080`

## 2. 健康检查

```bash
curl http://127.0.0.1:18080/health
```

结果：

```json
{"status":"ok"}
```

`/props` 返回的默认生成参数中包含：

```json
{"priority":100}
```

说明 priority 字段已经进入 server task 参数序列化路径。

## 3. 并发优先级测试

先启动低优先级长请求，再间隔 200 ms 启动高优先级短请求。

低优先级请求：

```json
{
  "prompt": "请连续输出从1到1000的数字，每个数字用逗号分隔。",
  "n_predict": 500,
  "priority": 10,
  "temperature": 0,
  "stream": false
}
```

高优先级请求：

```json
{
  "prompt": "只回答HIGH_DONE",
  "n_predict": 8,
  "priority": 0,
  "temperature": 0,
  "stream": false
}
```

请求执行结果：

| 请求 | slot | priority | 开始时间 | 结束时间 | tokens_predicted |
|---|---:|---:|---:|---:|---:|
| 低优先级长请求 | 1 | 10 | 1786168682078 ms | 1786168690446 ms | 500 |
| 高优先级短请求 | 0 | 0 | 1786168682284 ms | 1786168682543 ms | 8 |

高优先级请求在低优先级请求仍运行期间完成，端到端耗时约 259 ms；低优先级请求随后继续并完成 500 token，端到端耗时约 8.37 s。

服务日志顺序也符合预期：

1. slot 1 启动 task 0（低优先级）；
2. slot 0 启动 task 2（高优先级）；
3. slot 0 先完成并释放；
4. slot 1 继续生成并完成 500 token。

高优先级响应的 JSON 包含 `"id_slot":0`、`"tokens_predicted":8`；低优先级响应包含 `"id_slot":1`、`"tokens_predicted":500`、`"tokens_evaluated":21`。

## 4. 测试结论

本次测试验证了：

* 指定 HMM 模型能够正常加载并运行在 Houmo NPU 上；
* priority 请求字段可被解析并返回；
* 不同请求使用隔离 slot；
* 高优先级 slot 可以在低优先级 slot 仍未完成时优先执行并完成；
* 低优先级 slot 的状态未被清理，随后能够继续完成生成。

## 5. 测试限制

本次服务使用两个 slot，因此高优先级请求获得了空闲 slot。测试验证的是“跨 slot 调度优先级”，不是复用同一 slot 的上下文迁移。

## 6. 同优先级并发测试

服务启动时未使用 `--no-cont-batching`。同时提交两个 `priority=5`、各生成 80 token 的请求，结果如下：

| 请求 | slot | priority | 开始时间 | 结束时间 | tokens_predicted |
|---|---:|---:|---:|---:|---:|
| A | 1 | 5 | 1786169097370 ms | 1786169100105 ms | 80 |
| B | 0 | 5 | 1786169097372 ms | 1786169100120 ms | 80 |

两个请求几乎同时启动并分别使用 slot 1/0，说明同优先级 slot 不会互相抢占，可以并发执行。

当前实现的抢占边界是一次 HMM decode 调用返回之后；已经提交到 NPU 的单次 kernel 不会被 OS 线程强制中断。若所有 slot 都被低优先级请求占满，高优先级请求仍需等待可用 slot；要在这种情况下复用低优先级 slot，必须另行实现 slot 状态持久化和恢复。

## 7. 初次测试记录

第一次并发测试中，低优先级 slot 仍持续执行到结束后高优先级请求才完成。原因是调度器只过滤了 `SLOT_STATE_GENERATING`，没有过滤高优先级请求处于 prompt processing 阶段时的低优先级 slot。修正为 HMM 每轮只选择一个最高优先级活动 slot 后，第二次测试通过。
