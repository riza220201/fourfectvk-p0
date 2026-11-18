#!/usr/bin/env python3
"""Summarize a faultrun.sh trace: the probe's markers interleaved with the kbase kprobes, times relative to the
first marker (only "faultprobe:" marker lines; the power HAL writes there too). Contexts are ours if the probe dequeued events from them (kbase_event_dequeue runs in its read()).
Frequent per-atom probes are counted, rare ones (stops, faults, resets, teardown, soft-job cancels) listed.
usage: faultsum.py <trace> [--all]  (--all lists every probe hit for our contexts)"""
import collections, re, sys

LINE = re.compile(r'^\s*(?P<comm>.+?)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+\S+\s+(?P<t>\d+\.\d+): (?P<ev>[a-z_]+): (?P<rest>.*)$')
FREQ = {'pull', 'requeue', 'jmcomplete', 'jddone', 'doneworker', 'post', 'dequeue', 'stopslot', 'softstop'}

def main(path, show_all):
    rows = []
    for l in open(path, errors='replace'):
        if l.startswith('#'):
            continue
        m = LINE.match(l)
        if m:
            rows.append((float(m['t']), m['comm'].strip(), int(m['pid']), m['ev'], m['rest']))
    rows = [r for r in rows if r[3] != 'tracing_mark_write' or r[4].startswith('faultprobe: ')]
    marks = [r for r in rows if r[3] == 'tracing_mark_write']
    if not marks:
        print('no markers'); return
    t0 = marks[0][0]
    probe_pids = {r[2] for r in marks}
    ours = set()
    for t, comm, pid, ev, rest in rows:
        if ev == 'dequeue' and pid in probe_pids:
            ours.add(re.search(r'kctx=(0x[0-9a-f]+)', rest)[1])
    print(f'markers {len(marks)}, probe pid(s) {sorted(probe_pids)}, our kctx {sorted(ours)}')
    counts = collections.Counter()
    last_reset = -1e9
    for t, comm, pid, ev, rest in rows:
        if ev == 'resetworker':
            last_reset = t
        if ev == 'tracing_mark_write':
            print(f'{t - t0:10.6f}  -- {rest.replace("faultprobe: ", "")}')
            continue
        k = re.search(r'kctx=(0x[0-9a-f]+)', rest)
        mine = (k[1] in ours) if k else None
        counts[(ev, {True: 'ours', False: 'other', None: '-'}[mine])] += 1
        if mine is False:
            continue
        if ev in ('inithw', 'inithw_ret') and t - last_reset > 10 and not show_all:
            continue   # kbase_pm_init_hw also runs at every GPU power-up; list it only inside a reset
        if show_all or ev not in FREQ:
            print(f'{t - t0:10.6f}  {ev:12s} {comm}-{pid} {re.sub(r"^.*?\) ?", "", rest)}')
    print('counts (probe, kctx):')
    for (ev, who), n in sorted(counts.items()):
        print(f'  {ev:12s} {who:5s} {n}')

if __name__ == '__main__':
    main(sys.argv[1], '--all' in sys.argv)
