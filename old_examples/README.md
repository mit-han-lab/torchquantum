# Legacy Examples

Examples in this directory have **not** been validated against the current
torchquantum stack (qiskit >= 2.5) and are kept for reference only. Most of
them target retired APIs (`qiskit.pulse`, `IBMQ`, `qiskit.opflow`,
`BackendProperties`, legacy fake backends) or depend on packages that are no
longer part of the requirements (`cuquantum`, `tensorflow`).

| Directory | Why it is here |
| --- | --- |
| `ICCAD22_tutorial`, `QCE22_tutorial` | Conference tutorial notebooks written for qiskit < 1.0 (incl. removed `qiskit.pulse`) |
| `QuantumNAS` | Notebook targeting the legacy IBMQ account flow |
| `quest` | Research companion code using removed `BackendProperties` / legacy fake backends / `assemble` |
| `cuquantum` | Requires the optional `cuquantum` package |
| `quantum_kernel_method`, `gradient_pruning`, `quantum_transformer` | Notebook-only examples, not re-validated after the qiskit 2.5 upgrade |

Working, validated examples live in [`examples/`](../examples/).
