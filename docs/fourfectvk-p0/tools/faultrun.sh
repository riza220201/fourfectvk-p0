#!/system/bin/sh
# faultrun.sh <name> <faultprobe args...>: run tools/probes/faultprobe under kbase kprobes (trace_clock=mono, the
# probe's own clock), save the trace + the new kernel log lines to /data/local/tmp/mm/fault-<name>.{trace,dmesg,out},
# then remove every kprobe and restore trace_clock/buffer size. Run as root.
# FR_BIN=<binary> runs that instead of faultprobe (e.g. vkhang, through p0run.sh).
# FR_EXTRA="<kprobe line>;<kprobe line>...": more kprobes in group fp (removed with the rest).
# FR_JS_TIMEOUTS="<8 values>": kbase js_timeouts for the run (restored after), e.g. a reset before the hard-stop.
N=$1; shift
D=/data/local/tmp/mm; T=/sys/kernel/tracing
JST=/sys/class/misc/mali0/device/js_timeouts; OLDJS=$(cat $JST)
OLDCLK=$(sed -E 's/.*\[([a-z_]+)\].*/\1/' $T/trace_clock); OLDBUF=$(cat $T/buffer_size_kb)
cleanup() {
  echo 0 > $T/tracing_on; echo 0 > $T/events/fp/enable 2>/dev/null
  grep -E '^[pr][0-9]*:fp/' $T/kprobe_events | sed -E 's/^[pr][0-9]*:(fp\/[a-z_]+).*/-:\1/' | while read l; do echo "$l" >> $T/kprobe_events; done
  echo $OLDCLK > $T/trace_clock; echo $OLDBUF > $T/buffer_size_kb
  echo "$OLDJS" > $JST; echo "js_timeouts restored: $(cat $JST)"
}
trap cleanup EXIT INT TERM
echo 0 > $T/tracing_on; echo mono > $T/trace_clock; echo 8192 > $T/buffer_size_kb; echo > $T/trace
add() { echo "$1" >> $T/kprobe_events || echo "kprobe FAILED: $1"; }
add 'p:fp/pull kbase_js_pull kctx=%x0 js=%x1'
add 'p:fp/requeue kbase_jm_return_atom_to_js katom=%x1'
add 'p:fp/softstop kbase_job_slot_softstop_swflags js=%x1 katom=%x2'
add 'p:fp/hardstop kbase_job_slot_hardstop kctx=%x0 js=%x1 katom=%x2'
add 'p:fp/stopslot kbase_backend_soft_hard_stop_slot a1=%x1 a2=%x2 a3=%x3 a4=%x4'
add 'p:fp/jmcomplete kbase_jm_complete katom=%x1'
add 'p:fp/jddone kbase_jd_done katom=%x0'
add 'p:fp/doneworker kbase_jd_done_worker'
add 'p:fp/post kbase_event_post kctx=%x0 katom=%x1'
add 'p:fp/dequeue kbase_event_dequeue kctx=%x0'
add 'p:fp/mmuirq kbase_mmu_interrupt_process kctx=%x1 as=%x2'
add 'p:fp/pfworker kbase_mmu_page_fault_worker'
add 'p:fp/faultkill kbase_mmu_report_fault_and_kill kctx=%x0'
add 'p:fp/buskill kbase_gpu_report_bus_fault_and_kill kctx=%x0'
add 'p:fp/prepreset kbase_prepare_to_reset_gpu_locked'
add 'p:fp/resetlocked kbase_reset_gpu_locked'
add 'p:fp/reset kbase_reset_gpu'
add 'p:fp/resettimer kbasep_reset_timer_callback'
add 'p:fp/resetworker kbasep_reset_timeout_worker'
add 'p:fp/backendreset kbase_backend_reset'
add 'p:fp/inithw kbase_pm_init_hw'
add 'r:fp/inithw_ret kbase_pm_init_hw'
add 'p:fp/zap kbase_jd_zap_context kctx=%x0'
add 'p:fp/destroy kbase_destroy_context kctx=%x0'
add 'r:fp/destroy_ret kbase_destroy_context'
add 'p:fp/cancelsoft kbase_cancel_soft_job katom=%x0'
add 'p:fp/softtimeout kbasep_soft_job_timeout_worker'
add 'p:fp/jdcancel kbase_jd_cancel katom=%x1'
[ -n "$FR_JS_TIMEOUTS" ] && echo "$FR_JS_TIMEOUTS" > $JST && echo "js_timeouts for this run: $(cat $JST)"
if [ -n "$FR_EXTRA" ]; then
  echo "$FR_EXTRA" | tr ';' '\n' | while read l; do [ -n "$l" ] && add "$l"; done
fi
echo 1 > $T/events/fp/enable; echo 1 > $T/tracing_on
MARK="faultrun-$N-$(date +%s)"; echo "$MARK" > /dev/kmsg
echo "== faultrun $N: faultprobe $* (uptime $(cut -d' ' -f1 /proc/uptime))"
${FR_BIN:-$D/faultprobe} "$@" 2>&1 | tee $D/fault-$N.out
echo 0 > $T/tracing_on
cat $T/trace > $D/fault-$N.trace
dmesg | sed -n "/$MARK/,\$p" > $D/fault-$N.dmesg
echo "== trace lines $(grep -vc '^#' $D/fault-$N.trace), kernel log lines after the marker $(wc -l < $D/fault-$N.dmesg)"
