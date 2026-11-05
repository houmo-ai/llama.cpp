# HouMo runtime automatic recovery

HouMo TCIM runtime 1.4.0 can reset the IPU after an inference timeout while
preserving loaded modules, tensor bindings, and model caches. The HouMo backend
therefore retries only the failed synchronous `tcim::Module::Run(true)` call;
the enclosing decode or multi-round prefill operation is not replayed.

Recovery is process-wide and keyed by individual physical device IDs. Modules
whose device sets overlap share recovery state for the overlapping devices.
Normal inference remains concurrent; shared state only delays calls during a
recoverable-error backoff.

Status handling:

- `TIMEOUT`: TCIM has completed reset; retry immediately.
- `ERR_KERNEL`: wait briefly, then retry.
- `RESOURCE_EXHAUSTED`: wait briefly, then retry.
- `RECOVERING`: the runtime is still resetting; wait briefly, then retry.
- `ERR_FATAL`: do not retry; return failure and instruct the operator to run
  `hm_smi -rd`.
- Any other non-OK status: do not retry.

The backend defaults `HDPL_API_TIMEOUT` to `5000000` microseconds before the
first HouMo device manager is created. An explicitly configured value is never
overwritten.

The transient retry delay defaults to 100000 microseconds and can be changed
with `HOUMO_RUN_RETRY_SLEEP_US`. Recoverable statuses are retried until the run
succeeds or TCIM returns `ERR_FATAL`/another non-retryable status.
