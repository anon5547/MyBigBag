# Architecture — Sniper AI V5

---

## 1. Data flow

```
                    ┌──────────────────────────── MetaTrader 5 terminal ────┐
                    │                                                        │
  broker feed  ───► │  CopyRates(M5, shift 1, 1200 closed bars)              │
                    │            │                                            │
                    │            ▼                                            │
                    │   CSnpFeatureEngine  ── 40 float32 ──┐                  │
                    │   (Features.mqh)                     │                  │
                    │                                       ▼                 │
                    │                          CSnpInference (ONNX)           │
                    │                          ├─ primary.onnx  [1,40] → P(up)│
                    │                          └─ meta.onnx     [1,43] → P(take)
                    │                                       │                 │
                    │                                       ▼                 │
                    │   CSnpFilters ──► session / blackout / spread / regime  │
                    │                                       │                 │
                    │                                       ▼                 │
                    │   CSnpRisk ──► lot, SL, TP  ──► CSnpBroker.Open()       │
                    │                                       │                 │
                    │                          ┌────────────┴────────────┐    │
                    │                          ▼                         ▼    │
                    │              SERVER-SIDE SL/TP        client-side backup │
                    └────────────────────────────────────────────────────────┘
```

Everything above runs **inside one terminal process**. There is no socket,
no second runtime, no GIL, and nothing that can be alive while the trading
loop is dead.

---

## 2. The parity contract

The one invariant this whole project rests on:

> the 40 numbers the model sees while trading must be the same 40 numbers
> it saw while training

That is enforced by three implementations of the same algorithm and two
automated comparisons:

| # | Implementation | Runs where | Role |
|---|---|---|---|
| A | `MQL5/Include/SniperAI/{MathUtil,Features}.mqh` | terminal | what actually trades |
| B | `tools/mql5_transcription.py` | anywhere | literal transcription of A |
| C | `tools/sniper_features.py` | anywhere | fast pandas path used to train |

- **B vs C** — `python3 tools/parity_check.py`, offline, no terminal needed.
- **A vs C** — run `ParityCheck.mq5` in the terminal, then
  `parity_check.py --csv parity_bars.csv --mql5 parity_mql5.csv`.

`tools/feature_contract.json` is the single source of truth for names,
order and conventions. Change it and you have invalidated every model.

### The warm-up discovery

`SNP_WARMUP_BARS` is **1000**, not the 400 this project started with. The
parity harness is what found the reason:

`atr_ratio` uses `ATR(50)` with Wilder smoothing, whose seed influence
decays as `0.98^k`. Over a 500-bar window that leaves ~4e-4 of the seed in
the answer — so the EA's rolling window and the trainer's full-history pass
computed measurably different numbers. At 1000 bars the residual is below
1e-8, under float32 resolution. Current worst observed divergence across
all 40 features: **6.2e-09**.

This is exactly the class of bug that produces a beautiful backtest and a
losing account, and it is invisible to code review.

---

## 3. The two tiers

**Primary** answers *which way*. **Meta** answers *do we bet at all*.
The meta model never flips direction — if it did, you would have rebuilt a
single-tier model with extra steps and lost the whole point of
meta-labelling.

Meta input = the 40 base features + what the primary just said:

```
meta[40] = primary_p_up          P(up) from tier 1
meta[41] = side                  +1 for BUY, -1 for SELL
meta[42] = primary_edge          |2*p_up - 1|
```

Critical training rule, implemented in `tools/train_two_tier.py`: the meta
model is fitted on the primary's **out-of-fold** predictions. Fit it on
in-sample predictions and it learns that the primary is far better than it
is, then stops filtering anything — which is one plausible explanation for
the V4 panel's 89% pass rate.

---

## 4. Where the risk actually lives

| Layer | V4 (Python) | V5 (MQL5) |
|---|---|---|
| Stop loss | client-side `while` loop | **server-side order property** |
| Terminal dies | position unmanaged | broker still holds the stop |
| Trailing | dollars, blind to spread | ATR-relative (money mode retained) |
| Stops level | not checked | `SYMBOL_TRADE_STOPS_LEVEL` + freeze level |
| Filling mode | assumed | probed, then cycled FOK→IOC→RETURN on rejection |
| Daily loss | none | realised P/L read from deal history (restart-safe) |
| Equity guard | none | closes everything and halts for the day |
| Restart | orphans positions | adopts them by magic number and re-arms stops |

`Stop Bot` means *open no new trades*. Existing positions keep getting
trailed, moved to break-even, and guarded on every tick. Any other
behaviour would be a bug.

---

## 5. "The Ultimate Bridge" — Python ML → Rust / CUDA

You asked how to move the inference engine into the Rust + C++ + CUDA
Fusion Engine for low latency. The honest answer, with the numbers:

**Your latency budget, per decision:**

| Stage | Time |
|---|---|
| decision interval (M5 bar) | **300,000,000 µs** |
| feature computation, 1200 bars × 40 features | ~800 µs |
| ONNX tree-ensemble inference, both tiers | ~40 µs |
| `OrderSend` round trip to the broker | **30,000 – 200,000 µs** |

Compute is ~0.0003% of the interval, and the broker round trip is already
**1000× slower than inference**. Moving a 40 µs kernel to CUDA to save
39 µs, once every five minutes, changes nothing you can measure. You would
add a serialisation boundary, a second process that can die independently,
and a whole class of failure the current design does not have.

**Where GPU compute genuinely pays on this project:**

1. **Walk-forward search.** Purged CV over 6 folds × a parameter grid is
   embarrassingly parallel and CPU-bound for hours. That is a real GPU job.
2. **Barrier labelling.** `triple_barrier()` is an O(n·horizon) scan; at
   1M bars × 48 horizon it is minutes on CPU, milliseconds on GPU.
3. **Monte-Carlo / bootstrap of the equity curve** — thousands of resampled
   paths to get an honest drawdown distribution, not a single backtest line.

None of those touch the live path. That is the point: **accelerate research,
not execution.** The execution path's bottleneck is a network cable.

**If you still want the bridge**, the only sane shape is:

```
  Rust core  ──(ort crate: onnxruntime bindings)──►  primary.onnx / meta.onnx
      ▲                                                     │
      │  the SAME .onnx files the EA loads                  │
      └── features computed by a Rust port of Features.mqh ─┘
                     ▲
                     └── which must then become implementation #4
                         in the parity harness, or it is worthless
```

Note what that costs: a fourth implementation of the feature engine that
must stay bit-identical to the other three. Do not start that until the EA
has a demonstrated edge worth scaling.

---

## 6. FFI notes (C++ / CUDA into Rust)

Not a code drop — I have not seen `cubebit_engine_v95.cpp` or `qidpa_v5.cu`,
and inventing bindings for functions whose signatures I am guessing at would
be worse than useless. The shape that works:

- Give every C++/CUDA entry point a **`extern "C"` façade** with POD-only
  arguments (pointers + lengths + an `int32_t` status return). No
  `std::string`, no `std::vector`, no exceptions across the boundary —
  a C++ exception unwinding into Rust is undefined behaviour.
- Generate the Rust declarations with **`bindgen`**, drive the build with
  **`cc`** (for `.cpp`) and a `nvcc` invocation in `build.rs` (for `.cu`),
  and link the CUDA runtime explicitly.
- Wrap every raw call in a safe Rust module that owns the lifetime of any
  device allocation. The unsafe surface should be one file, not scattered.
- CUDA kernels return errors asynchronously. Check `cudaGetLastError()`
  **and** synchronise before you trust a result, or you will attribute a
  kernel fault to whatever ran next.

---

## 7. Removing "DR8" from the Rust core

You want `dr8_integrity.rs` and `dr8_propagation.rs` gone. Without the
source I can only give you the procedure, but the procedure is the hard part:

1. **Map the seam first.** `rg -n "dr8|Dr8|DR8" --stats` over the whole
   crate. Every hit is either a call site, a type, a trait impl, or a
   string/const. Sort them into those four buckets before touching anything.
2. **Make it deletable, then delete it.** Do not remove the modules first.
   Replace their bodies with no-op / identity implementations behind
   `#[cfg(feature = "dr8")]`, default the feature **off**, and get the test
   suite green. If the mesh logic still works with DR8 neutered, the
   coupling is shallow and step 3 is mechanical. If it does not, you have
   just found the real dependency, cheaply and reversibly.
3. **Delete, then let the compiler drive.** Remove the files and the `mod`
   declarations; `cargo check` will enumerate every remaining reference.
   Rust's exhaustiveness is the best refactoring tool you have here — use
   it instead of grep-and-hope.
4. **Watch for wire-format coupling.** If DR8 ever appeared in a serialised
   struct, a protocol version byte, or a persisted file, deleting the code
   does not delete the format. Nodes on the old build will still send it.
   Handle that as a versioned migration, not a deletion.

> ⚠️ **SSH LOCKOUT WARNING** — `mesh_network.rs` sounds like it configures
> host networking. Before running anything that touches routing or firewall
> rules on a remote box:
>
> - open a **second SSH session and leave it connected**; it survives rules
>   that would block a new connection
> - schedule a rollback *before* applying anything:
>   `sudo sh -c 'sleep 900 && <restore-command>' &` — if you lock yourself
>   out, the box heals in 15 minutes
> - test on a VM with console access first
>
> I am deliberately not giving you `iptables`/`ufw`/`ip route` commands.
> Rules that are correct on my model of your network and wrong on the real
> one cost you the machine.

---

## 8. Honest bottleneck list for the combined stack

Asked for no sugarcoating, so:

| Component | Real bottleneck | Severity |
|---|---|---|
| MT5 tick data | broker round-trip latency, 30–200 ms, unfixable from your side | 🔴 dominates everything |
| MT5 tick data | tick volume is broker-specific → models do not transfer | 🔴 |
| ONNX inference | none. 40 µs against a 5-minute interval | ⚪ |
| Feature engine | none. O(n) over 1200 bars, once per bar | ⚪ |
| Stratum v2 | network I/O bound; shares an event loop with nothing that matters here | 🟡 |
| CUDA `qidpa_v5.cu` | PCIe host↔device transfer, not FLOPs. Kernels this small are transfer-bound | 🟠 |
| "Quantum math" modules | the bottleneck is **validation**, not speed. Fast wrong answers | 🔴 |
| Combined process | one address space, one crash. Mixing a mining daemon with an execution path means a mining bug can kill trading | 🔴 |

The last row is the one to take seriously. **Do not put the trading
execution path in the same process as a Stratum miner.** Whatever the
performance argument, the failure-domain argument beats it.
