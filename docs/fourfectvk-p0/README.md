# FourFectVK P0 — a bounded terminal acknowledgment on stock kbase JM

For Noysz (FourFectVK, `github.com/Noysz/panvk-g99-jm`). Your November status keeps T06 (memory) and T08
(overlap) blocked on one statement:

> Stock kbase has no verified supported bounded terminal mechanism that covers the required failure paths
> while keeping backing valid. Timeout, DEVICE_LOST, POST_TERM, fd close and MMU flush are not terminal
> acknowledgments.

We measured what stock r54p1 kbase does on every failure path, found a mechanism that *is* a terminal
acknowledgment, implemented it in PanVK, and proved it on hardware through the Vulkan API — on your November
tree, in your Android packaging. Everything needed to check it is in this repository.

## The branches

The root commit is upstream Mesa `feeadb5f` (2026-10-05) as a single snapshot, so the repository stays small.
Every later tree is byte-identical to the original history; **tree hashes are the reference**.

| branch | what | tree |
|---|---|---|
| `fourfectvk-november` | your 45-patch November candidate, reconstructed bit-exact from your manifest (`prepare_source.py --verify-with` passed) | `87622157` = your candidate |
| **`fourfectvk-p0`** (default) | `fourfectvk-november` + F1 (2 commits) + P0 (3 commits) + this directory | `97cf1178` before the docs commit |
| `mesa-mobile` | our full tree for the itel RS4: your November tree + GLES on kbase (gallium panfrost) + Android/RS4 fixes + P0 — what ships in our AICP build 23 | `4c10e3c9` |

```
git diff fourfectvk-november fourfectvk-p0~1 -- src build-support   # F1 + P0 only
git format-patch fourfectvk-november..fourfectvk-p0~1                # as a series for patches-candidates/
```

**F1 (2 commits, needed to build and package the series):** your patch 0001 deletes upstream's `build-support/`;
meson's `version-script`/`dynamic-list` link probes read `build-support/conftest.{map,dyn}`, both probes fail and
**test9 ships with no export map** (verified on your release asset `492816fb…`: 335 defined dynamic exports,
331 of them statically linked SPIRV-Tools). `build: restore build-support/` + `panvk: Android export map with
HMI and the vk_icd entry points` give exactly HMI + the 3 `vk_icd*` entry points (your harnesses and the CTS
vkshim dlsym `vk_icdGetInstanceProcAddr`, so both sets are exported).

**P0 (3 commits):** `panvk/kbase: P0 -- hold freed memory after a JM failure until a barrier proves the GPU
done`, `settle deadline from the last kbase event; name JOB_CANCELLED`, `watchdog opt-in`. 3 files,
+389/−26: `src/panfrost/lib/kmod/kbase_kmod.{c,h}`, `src/panfrost/vulkan/jm/panvk_vX_gpu_queue.c`.

## What stock r54p1 JM kbase does (measured)

Device: itel RS4 (MT6789, Mali-G57 MC2), kbase `r54p1-12eac0` (UK 11.46), Linux 5.10, from the same Samsung
A175F `a16w_jm` source family as your Note 40's module. Tool: `tools/faultprobe.c` — raw kbase ioctls, no Mesa
— under ~30 kprobes on kbase (`tools/faultrun.sh`, `trace_clock=mono` = the probe's clock, so kernel and
userspace events share one timeline). Probe outputs: `results/fault-*.out`.

| failure | what kbase does | bounded? |
|---|---|---|
| **MMU fault** among in-flight atoms (write to a freed BO's VA; dmesg TRANSLATION_FAULT at exactly that VA) | the faulting atom → **TERMINATED** (not the MMU code); its DATA dependents TERMINATED; **every other atom of that context still in flight — ORDER-dependent and independent — JOB_READ_FAULT, never run**; every atom exactly one event, all within ~1.1 ms; other contexts unaffected; the context is reusable at once | yes, ~1 ms |
| **Infinite-loop shader** (hand-assembled Valhall: `IADD_IMM.wait0; STORE.i32; BRANCHZ.reconverge 1.0, offset:-3`) | one soft-stop at +200 ms cannot take it off (the workgroup never ends); **hard-stop at 5.107 s → TERMINATED 0.6 ms later**; the GPU had stopped before the event (freeing the store target right after: no fault); context reusable | yes, ~5.1 s |
| **The same with the reset path forced** (`js_timeouts` reset 2 s < hard-stop 30 s, so it resets with the shader still on the GPU) | JOB_CANCELLED at 5.12 s (2 s + `reset_timeout` 3 s) | yes, ~5.1–8.5 s |
| **An endless job chain that obeys soft-stop** (2 WRITE_VALUE jobs clearing each other's status, Suppress Prefetch) | **never hard-stopped**: soft-stopped every ~75 ms, requeued, pulled again with `katom->ticks = 0` (mali_kbase_js.c:2646) — 401 pulls in 30 s, no event; it also starves every later same-priority atom of its context on that slot | **no** — only a broken (cyclic) job chain makes one, i.e. a driver bug |
| **FENCE_WAIT on a foreign fence that never signals** | no event in 60 s (not cancelled: only EVENT_WAIT has `soft_job_timeout`, control: cancelled at 3.03 s); tearing down the *producer* signals the fence −EFAULT and the waiter is cancelled 0.03 ms later | **no** |

Three facts decide the design:

1. **A JOB_CANCELLED/TERMINATED event is not, by design, a terminal acknowledgment.** In 25/25 forced-reset runs
   the event was posted after `kbase_pm_init_hw` (the GPU soft reset) returned, but only because the done worker's
   post is deferred to its end and it happens to stall behind the reset worker. Nothing in the completion path is
   guaranteed to wait for the reset: `runpool_mutex` is taken by the reset worker only after `init_hw`, and
   `pm.lock` (held across `init_hw`) only conditionally, when the scheduler finds another inactive context to
   schedule. In 9/15 runs the worker had already started during the reset.
2. **An exported out-fence (FENCE_TRIGGER) is not one either.** `kbase_jd_done_nolock` resolves dependents
   *before* that deferred post, and a dependent FENCE_TRIGGER signals its sync_file there
   (`kbase_finish_soft_job` → `kbase_sync_fence_out_trigger(-EFAULT)`). Measured: the trigger finished 373 µs
   before the GPU reset completed, with the hung shader still on the GPU; through PanVK, 3 of 6 runs signalled
   during the reset, one of them seen by userspace 4.8 ms before a 5.1 ms reset finished.
3. **fd close is not an ordering point** (you are right): every BO mapping holds the kbase file, so the context
   (and the zap of its jobs) is destroyed only after the last unmap — the memory is freed first.

And one that *is* a terminal acknowledgment:

4. **A new GPU job completing after the failed work.** kbase starts nothing on the GPU while a reset is pending
   (`kbase_backend_slot_update` returns while `kbase_reset_gpu_is_active`), and a job queued behind the failed work
   runs after it. Measured (`faultprobe f5`, 8 forced resets): hang → ORDER → WRITE_VALUE barrier → DATA →
   FENCE_TRIGGER: **8/8 the barrier completed on the GPU after `kbase_pm_init_hw` returned**, including runs where the
   hang's dependents were resolved mid-reset (traced in detail: `f5r6`, resolved at 455 µs of a 213→579 µs reset;
   kbase had queued the barrier in the slot ringbuffer behind the hang at submit time and re-ran it after the reset); the fence behind it signalled after it, status 1.

## P0 as implemented

- **kbase kmod:** `kbase_kmod_quarantine_hold/release` on your existing quarantine list (the
  `PANVK_DBG_FREE_DELAY_MS` one): while held, a freed BO stays mapped on the GPU (entry marked `held`; the
  time-based reaper skips it).
- **JM queue:** every loss (a failed atom, a read or submit failure, the watchdog) goes through
  `panvk_kbase_mark_lost`: hold the quarantine and start a settle thread, which
  1. reads events until every in-flight submission's last atom (transitively all of its atoms: DATA chain, the
     wait by ORDER) and every export trigger has reported — terminal accounting continues after the loss without
     releasing atom numbers or advancing success seqnos (your 0003 semantics are kept);
  2. submits a WRITE_VALUE barrier (reserved atom 255, its own slot in a 32 KiB BO) and waits for it to report
     DONE (3 tries);
  3. releases the quarantine. Deadline `PANVK_KBASE_P0_DEADLINE_MS` (12 s) from the last kbase event; work that
     never ends cannot settle and keeps its memory held until device destroy — neither unbounded class has a GPU
     reset in flight, so freeing then faults on the GPU instead of reaching freed pages (measured: a running
     endless chain takes a translation fault on its unmapped descriptor at teardown).
- **Exports:** the FENCE_TRIGGER waits for a WRITE_VALUE barrier ORDER-dependent on the submission, so an exported
  sync_file never signals while the GPU may still run the work. It then reports success even when the work failed
  (the loss is reported through Vulkan). Cost: one tiny job per export, ~7 µs median.
- **Watchdog (opt-in):** a CPU wait with work in flight and no kbase event for `PANVK_KBASE_WATCHDOG_MS` latches a
  loss — the only way out of the unbounded classes for an application that waits forever. Off by default: a
  queued submission can legitimately wait longer than any fixed time (screen off: the compositor keeps the
  swapchain image it waits for), and a false DEVICE_LOST kills the app.

| env | default | |
|---|---|---|
| `PANVK_KBASE_P0` | 1 | 0 = everything above off (the control) |
| `PANVK_KBASE_EXPORT_BARRIER` | = P0 | the barrier before export FENCE_TRIGGERs |
| `PANVK_KBASE_P0_DEADLINE_MS` | 12000 | settle deadline, from the last kbase event |
| `PANVK_KBASE_WATCHDOG_MS` | 0 (off) | |

PanVK logs, one line per loss: `panvk: P0 settled 0.9 ms after the loss: N submission(s) in flight reported
after …, barrier DONE after … (1 try); released K freed BO(s), … KiB` — or `P0 NOT settled … (reason)`.

## Proof through the Vulkan API

`tools/vkhang.c` (+ `tools/shaders/hang.comp`: `while (true) atomicAdd(v[0], 1u);`) through the platform Vulkan
loader; `tools/p0run.sh` binds the driver under test as the HAL in a private mount namespace (nothing else on the
phone sees it) and runs it under the kprobes plus `kbase_gpu_munmap` — the moment kbase takes a BO off the GPU.
`tools/p0sum.py`, `resetwin.py`, `barrierwin.py` turn a trace into a timeline.

| run (results/) | path | result |
|---|---|---|
| `p0hang*`, `b23hang` | hard-stop | DEVICE_LOST 5.10 s; the app frees at once → **no `kbase_gpu_munmap`**; barrier done on the GPU +0.17–0.36 ms; held BOs unmapped +0.66–1.12 ms; settled 0.9–1.3 ms after the loss |
| `p0hangctl` | `PANVK_KBASE_P0=0` | freed at once; "retirement unproven at destroy" |
| `p0reset`, `b23export` | forced reset (JOB_CANCELLED, "Resetting GPU") | frees held across the reset; barrier on the GPU and unmaps after `kbase_pm_init_hw` returned |
| `p0multi*`, `b23multi` | hang + 5 queued submissions | "5 submission(s) in flight reported", then the barrier; settled ~1 ms |
| `p0x1r1–6` vs `p0x0r1–6` | export under forced reset, barrier on / off | **on: 6/6 status 1, signalled after the reset; off: 6/6 −14, during the reset in 3/6** |
| `p0perf*`, `b23perf` | 300 × tiny dispatch + SYNC_FD export | on 86–91 µs vs off 79–80 µs median; 0 bad |
| `p0wd`, `p0wd2` | work waiting on a never-signalled foreign fence (watchdog 5 s / 30 s) | `vkWaitForFences(UINT64_MAX)` → DEVICE_LOST at 5.006 / 30.03 s; "P0 NOT settled"; `vkDestroyDevice` 1.1 ms (`p0wd` waited 12 s: fixed by counting the deadline from the last kbase event) |
| **`nov-hang`, `nov-export`, `nov-multi`, `nov-perf`** | **your tree (`fourfectvk-p0`), your packaging (`VARIANT=noysz`, `libvulkan_panfrost.so` `803b2b09…`) bound as the HAL** | settled 0.5 / 1.1 / 1.2 ms; export status 1 under a forced reset; 5 queued reported; perf median 88 µs, 0 bad |

`b23*` = the same `mesa-mobile` P0 driver shipped as the system Vulkan driver of our AICP build 23 (no namespace),
which passes our full ROM audit; 32-bit (`vkfill32`) passes too.

## Not covered yet — please check on your side

- **Your device.** Our RS4 runs kernel 5.10; your Note 40 runs GKI 6.12 with the stock r54p1 module. The kbase
  code paths above are the same r54p1 source, but kprobe symbol availability and timings may differ: the tools
  print every step, so a run on your phone answers it. (The RS4's kbase carries one unrelated 4-byte change, a
  NOP'd store in MediaTek's `mtk_common_cal_gpu_utilization_ex` for the GED queue fence.)
- A Vulkan-level MMU fault (bad buffer device address), a CTS regression run on `fourfectvk-p0`, the overlap
  path (0012) with P0, process death during a hang (the kernel's `exit_mmap` unmaps before the fd closes —
  not fixable in userspace).

## Reproducing

```
# NDK r27 (aarch64-linux-android31)
clang -O1 -o faultprobe tools/faultprobe.c -lm
clang -O1 -I tools -o vkhang tools/vkhang.c -lvulkan     # shaders/*.inc: glslc -O --target-env=vulkan1.1 -mfmt=c
# on the phone, root (tracefs + kprobes), files in /data/local/tmp/mm:
sh faultrun.sh f4 f4 30                                   # raw kbase: infinite shader under kprobes
FR_JS_TIMEOUTS="100 100 30000 30000 1500000 2000 2000 1502000" sh faultrun.sh f5 f5 30   # barrier under a forced reset
P0LIB=/data/local/tmp/mm/libvulkan_panfrost.so sh p0run.sh hang hang   # PanVK through Vulkan
```
`faultrun.sh` restores `js_timeouts`, `trace_clock` and the buffer size and removes every kprobe it added.
`p0run.sh` needs busybox `unshare` (ours: KernelSU's) and `chcon` to `same_process_hal_file`.

Tools and docs here: MIT, like Mesa. Your patches are yours; the `fourfectvk-november` branch is your
series applied as one commit per patch.
