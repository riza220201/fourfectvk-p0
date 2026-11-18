#!/system/bin/sh
# p0run.sh <name> <vkhang args...>: run tools/probes/vkhang under faultrun.sh's kbase kprobes (+ kbase_gpu_munmap,
# the moment kbase takes a BO off the GPU), with the PanVK under test (/data/local/tmp/mm/p0/vulkan.panvk.so) bound
# over /vendor/lib64/hw in a private mount namespace: nothing outside this process tree sees it. PanVK's own log
# lines (logcat, from the run's start) go to fault-<name>.logcat. Env (PANVK_KBASE_*) passes through. Run as root.
# P0LIB=<path> binds another build as the HAL (e.g. the FourFectVK package libvulkan_panfrost.so).
M=/data/local/tmp/mm
if [ "$1" != inner ]; then
  exec /data/adb/ksu/bin/busybox unshare -m --propagation private sh "$0" inner "$@"
fi
shift
N=$1; shift
H=$M/hw64p0; rm -rf $H; mkdir -p $H
cp -a /vendor/lib64/hw/. $H/
cp ${P0LIB:-$M/p0/vulkan.panvk.so} $H/vulkan.panvk.so; chmod 644 $H/vulkan.panvk.so
chcon u:object_r:same_process_hal_file:s0 $H/vulkan.panvk.so
mount --bind $H /vendor/lib64/hw || { echo "bind failed"; exit 1; }
echo "== p0run $N: /vendor/lib64/hw/vulkan.panvk.so $(sha256sum /vendor/lib64/hw/vulkan.panvk.so | cut -c1-16) (system's: $(sha256sum $H/../hw64p0/../p0/vulkan.panvk.so | cut -c1-16))"
START=$(date '+%m-%d %H:%M:%S.000')
FR_BIN=$M/vkhang FR_EXTRA="p:fp/gpumunmap kbase_gpu_munmap kctx=%x0${FR_EXTRA:+;$FR_EXTRA}" sh $M/faultrun.sh $N "$@"
logcat -d -v monotonic -T "$START" 2>/dev/null | grep -E "panvk|MESA|kbase" > $M/fault-$N.logcat
echo "== PanVK log ($(wc -l < $M/fault-$N.logcat) lines):"; grep -E "P0|lost|watchdog|failed|unproven" $M/fault-$N.logcat | tail -8
