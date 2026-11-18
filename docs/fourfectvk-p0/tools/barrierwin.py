#!/usr/bin/env python3
"""barrierwin.py <f5 traces>: per run, relative to the reset worker (us): the GPU reset (kbase_pm_init_hw in the reset
worker), when the hang's dependents were resolved (first kbase_jd_done_nolock in a kworker), when our context was
next pulled onto the GPU (the barrier), the barrier's hardware completion (kbase_jd_done after that pull), the
FENCE_TRIGGER's completion (kbase_finish_soft_job) and when the probe saw the fence. Safe = the barrier completed on
the GPU after the reset returned (it may sit in the slot's ringbuffer from submit time)."""
import re, sys
LINE = re.compile(r'^\s*(?P<comm>.+?)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+\S+\s+(?P<t>\d+\.\d+): (?P<ev>[a-z_]+): (?P<rest>.*)$')
for path in sys.argv[1:]:
    rows = [(float(m['t']), m['comm'].strip(), int(m['pid']), m['ev'], m['rest'])
            for m in (LINE.match(l) for l in open(path, errors='replace')) if m]
    probe = {r[2] for r in rows if r[3] == 'tracing_mark_write' and r[4].startswith('faultprobe: ')}
    ours = {re.search(r'kctx=(0x[0-9a-f]+)', r[4])[1] for r in rows if r[3] == 'dequeue' and r[2] in probe}
    rw = [r for r in rows if r[3] == 'resetworker']
    if not rw:
        print(path, 'no reset'); continue
    t0, wpid = rw[0][0], rw[0][2]
    us = lambda t: (t - t0) * 1e6
    w = [r for r in rows if t0 <= r[0] <= t0 + 0.02]
    ih = [us(r[0]) for r in w if r[3] == 'inithw' and r[2] == wpid][:1]
    ihr = [us(r[0]) for r in w if r[3] == 'inithw_ret' and r[2] == wpid][:1]
    dn = [us(r[0]) for r in w if r[3] == 'w_donenl' and 'kworker' in r[1]][:1]
    posts = [re.search(r'katom=(0x[0-9a-f]+)', r[4])[1] for r in rows if r[3] == 'post'
             and re.search(r'kctx=(0x[0-9a-f]+)', r[4])[1] in ours]
    hang = posts[0] if posts else None
    # the barrier = one of our posted atoms, not the hang, that completed on the hardware (kbase_jd_done)
    jd = [us(r[0]) for r in w if r[3] == 'jddone' and re.search(r'katom=(0x[0-9a-f]+)', r[4])[1] in posts[1:]][:1]
    pu = jd
    ff = [us(r[0]) for r in w if r[3] == 'f_finish'][:1]
    fs = [us(r[0]) for r in w if r[3] == 'tracing_mark_write' and 'FENCE signalled' in r[4]][:1]
    g = lambda l: f'{l[0]:.0f}' if l else '-'
    during = dn and ihr and dn[0] < ihr[0]
    safe = pu and ihr and pu[0] > ihr[0]
    print(f'{path.split("/")[-1]}: reset {g(ih)}->{g(ihr)}  dependents resolved {g(dn)}{" (DURING reset)" if during else ""}'
          f'  barrier done on the GPU {g(jd)}  trigger {g(ff)}  fence seen {g(fs)}'
          f'  => barrier ran {"AFTER" if safe else "BEFORE/UNKNOWN"} the reset')
