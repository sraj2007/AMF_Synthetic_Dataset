# SCOPE-5G: A Standards-Aligned Synthetic Dataset for AMF Performance Monitoring

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**SCOPE-5G** is a standards-aligned synthetic dataset of 5G Access and Mobility Management Function (AMF) performance measurements, generated under a nine-layer framework that combines 3GPP TS 28.552 measurement semantics, M/M/c queueing for resource modeling, fractional Gaussian noise for long-range temporal dependence, and sigmoid-ramped anomaly injection. The released dataset comprises 5,760 fifteen-minute observation slots over 60 days for a single AMF instance serving 100,000 user equipments (70% eMBB, 20% mMTC, 10% URLLC), with 162 labeled anomaly slots (2.8%) covering eight fault types arranged in three balanced cycles. SCOPE-5G is intended for fault-detection research, anomaly-detection benchmarking, and methodological studies of synthetic-data fidelity; it is not a substitute for real operational traces, and validation boundaries are documented in §X of the companion paper.

## Quick Start

```python
import pandas as pd
df = pd.read_csv('amf_synthetic_dataset.csv', parse_dates=['timestamp'])
print(df.shape)              # (5760, 112)
print(df['is_anomaly'].sum()) # 162
```

Open `AMF_Pipeline_Complete.ipynb` in Jupyter, Colab, or VS Code and run all cells to regenerate the dataset from scratch under `seed=42` (deterministic, ~30–60 seconds).

## Dataset Schema (`amf_synthetic_dataset.csv`)

The released CSV contains **112 columns** organized as follows. Of these, **102** are native TS 28.552 measurements; **6** are derived indicators (Jackson per-slice throughput estimates and SLA breach flags) flagged for users who want to exclude them; and **4** are metadata.

### Metadata (4 columns)

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Observation slot start (15-min granularity) |
| `amf_instance_id` | str | Instance identifier (e.g. `AMF_00`) |
| `is_anomaly` | int | 1 if the slot falls within an anomaly window, else 0 |
| `anomaly_type` | str | Fault type label (one of 8 types) or empty for normal slots |

### Native TS 28.552 measurements (102 columns)

Grouped by 3GPP measurement family:

| Family | Prefix | Count | Examples |
|---|---|---|---|
| Registration Management | `RM.` | 12 | `RM.RegReqAtt`, `RM.RegReqSucc`, `RM.RegSuccRate`, `RM.InitRegReqAtt` |
| Connection Management | `CM.` | 5 | `CM.ServiceReqAtt`, `CM.N2RelAtt`, `CM.ServiceReqSuccRate` |
| Mobility Management | `MM.` | 12 | `MM.HoReqAtt`, `MM.XnHoReqAtt`, `MM.HoSuccRate` |
| Paging | `PAG.` | 6 | `PAG.PagingReqAtt`, `PAG.PagingReqSucc`, `PAG.PagingDiscarded` |
| UE Context | `UC.` | 6 | `UC.UeContextCreated`, `UC.UeContextReleased`, `UC.ActiveUeContext` |
| Session Management | `SM.` | 10 | `SM.PduSessEstabAtt`, `SM.PduSessEstabSuccRate` |
| Authentication | `AUTH.` | 6 | `AUTH.AuthProcAtt`, `AUTH.AuthProcSucc`, `AUTH.NasSecModeAtt` |
| N1/N2 NAS | `N1N2.` | 6 | `N1N2.N1MsgSent`, `N1N2.N2MsgRecv`, `N1N2.NgapConnActive` |
| Resource / Performance | `RES.` | 39 | `RES.CpuUtil`, `RES.MemUtil`, `RES.Latency_ms`, per-slice CPU/Mem/Lat |

The full column list and 3GPP TS 28.552 / TS 23.502 cross-references are documented in Tables 4–7 of the companion paper.

### Derived indicators (6 columns)

Kept in the CSV for convenience but excluded from native-feature evaluations (per the LSTM-AE/IF/OCSVM detector pipelines):

- `RES.Jackson_eMBB_eps`, `RES.Jackson_mMTC_eps`, `RES.Jackson_URLLC_eps` — per-slice throughput estimates from a Jackson queueing network model
- `SLA.eMBB_breach`, `SLA.mMTC_breach`, `SLA.URLLC_breach` — boolean SLA breach flags derived from latency thresholds

## Anomaly Catalog

Eight fault types, three cycles of eight scenarios each = 24 total fault windows over 60 days:

| Type | Behavior |
|---|---|
| `cpu_overload` | Sustained CPU resource saturation (sigmoid ramp + plateau) |
| `memory_leak` | Gradual memory growth with associated latency degradation |
| `registration_storm` | Anomalous spike in registration attempts |
| `signaling_storm` | Surge in N1/N2 message volume |
| `handover_failure` | Elevated handover failure rate |
| `ddos_fake_registrations` | High-volume fake registration attempts; low success rate |
| `slice_isolation_failure` | Cross-slice resource contention |
| `amf_overload_cascade` | Combined CPU/memory/latency degradation |

Each fault uses a sigmoid intensity ramp; details and ground-truth parameters are documented in the companion paper §VI.

## Companion Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| `AMF_Pipeline_Complete.ipynb` | Self-contained generator + validation pipeline. Regenerates `amf_synthetic_dataset.csv` from scratch, runs Tier 1–3 statistical validation against eight independent references, performs the ablation study, and emits all figures and `Table 11` from the paper. **Run this first.** | ~5 min |
| `AMF_Demo_MultiInstance.ipynb` | Multi-instance demonstration (N_AMF=3 over 14 days). Loads the generator class from the umbrella notebook and shows per-instance load balance, anomaly scoping (CPU-overload confined to `AMF_01`), and cross-instance correlation analysis. Produces the numbers for §VII.F of the paper. | ~1 min |
| `AMF_Detector_LSTMAutoencoder.ipynb` | Self-contained LSTM-autoencoder fault detector. Trains on overlapping 96-slot windows of the 102 native columns and evaluates under two splits: default clean-baseline (S1) and temporal-generalization (S2). Reproduces the LSTM-AE rows of Table 10 in §VII.D. | ~5 min on GPU, ~30 min on CPU |
| `AMF_Baseline_GANDownstream.ipynb` | Self-contained GAN-baseline downstream comparison. Fits TVAE and CTGAN synthesizers on the SCOPE-5G clean training rows, then retrains Isolation Forest and One-Class SVM on the regenerated data and evaluates against the original labeled anomaly slots. Substantiates the §VII.D claim that procedure-aware generation provides downstream utility beyond distributional fidelity. | ~10–20 min |

All notebooks are deterministic under `seed=42`; small variations across hardware are expected for TensorFlow components (LSTM-AE training, TVAE/CTGAN fitting).

## How to Reproduce the Paper's Results

1. Clone or download this repository.
2. Open `AMF_Pipeline_Complete.ipynb` in Jupyter, Google Colab, or VS Code. **Restart Kernel and Run All Cells.** This regenerates `amf_synthetic_dataset.csv` and emits every figure and Table 11 entry in the paper.
3. Run `AMF_Demo_MultiInstance.ipynb` to reproduce the §VII.F multi-instance numbers.
4. Run `AMF_Detector_LSTMAutoencoder.ipynb` to reproduce the LSTM-AE rows of Table 10 (§VII.D).
5. Run `AMF_Baseline_GANDownstream.ipynb` to reproduce the GAN-baseline downstream comparison (§VII.D).

## Validation Summary

SCOPE-5G has been statistically validated against eight independent references. Acceptance criterion: MAPE < 30% on min-max-normalized shape comparisons (with one row at 30.57% flagged in Table 11). Detailed metrics and references are tabulated in §VIII (Table 11) of the paper; a full list of statistical validation outputs is produced by `AMF_Pipeline_Complete.ipynb` and saved in the validation/ subdirectory of the output archive.

## Scope and Boundaries

SCOPE-5G models the **aggregate TS 28.552 AMF measurement layer** and is appropriate for:
- Anomaly-detection research on AMF telemetry
- Methodological studies comparing detector families
- Synthetic-data fidelity research
- Time-series forecasting on cellular control-plane signals

The dataset does **not** model:
- Container orchestration dynamics (autoscaling, pod restarts, configuration rollouts)
- Cross-Network-Function feedback (cascading SMF/UPF/UDM interactions beyond per-procedure success probabilities)
- Real-time data-plane traffic (it is a control-plane / AMF KPI dataset, not user-plane)

See §X of the companion paper for the full discussion of validation constraints and generalization boundaries.

## Citation

If you use SCOPE-5G, please cite the companion paper:

```bibtex
@article{araujo2026scope5g,
  title   = {{SCOPE-5G}: A Standards-Aligned Synthetic {AMF} Observability
             Dataset for Reproducible {5G} Core Benchmarking},
  author  = {Ara{\'u}jo Jr., Silvio R. and Kamienski, Carlos A. and
             Bianchi, Reinaldo A. C.},
  journal = {IEEE Access},
  year    = {2026},
  volume  = {<TBD>},
  pages   = {<TBD>},
  doi     = {<TBD-on-acceptance>}
}
```

(Update `volume`, `pages`, and `doi` with the IEEE Access values once the paper is assigned them on acceptance.)

## License

The dataset and accompanying code in this repository are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. You are free to share and adapt the material for any purpose, including commercially, provided that you give appropriate credit (cite the paper above), provide a link to the license, and indicate if changes were made. Full license text: https://creativecommons.org/licenses/by/4.0/

## Contact

For questions, errata, or bug reports, open an issue on this repository or contact the corresponding author via the email address in the companion paper.
