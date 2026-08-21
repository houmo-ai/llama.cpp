# llama-threaded-server

Custom in-process router for the HouMo backend. The standard `llama-server`
subprocess router is unchanged.

Each loaded preset owns an independent `llama_model`, `server_context`, TCIM
Module set, KV cache, and inference thread. HouMo models loaded from the same
canonical model path on the same device topology use a shared TCIM
`WeightManager`, so their device weights are shared while context state remains
isolated.

Build and run:

```bash
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-threaded-server -j
./build/bin/llama-threaded-server --models-preset models.ini
```

This executable is router-only. Use `--models-dir` or `--models-preset`; direct
single-model `-m` mode remains provided by `llama-server`.

An editable preset is available at:

`tools/threaded-server/postmo-models-preset.example.ini`
