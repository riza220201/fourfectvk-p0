#!/usr/bin/env python3
"""resetwin.py <faultrun trace>...: the reset window of an f4 race run. Times in microseconds relative to the reset
worker's start: backend reset (atoms on the GPU cancelled), kbase_pm_init_hw entry/return inside the reset worker
(the GPU soft reset), the done worker, our event's post and our read(). post-before-reset-return = the race."""
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
    ev = {}
    for t, comm, pid, e, rest in rows:
        if t < t0 or t > t0 + 0.05:
            continue
        k = re.search(r'kctx=(0x[0-9a-f]+)', rest)
        if e in ('inithw', 'inithw_ret') and pid != wpid:
            continue
        if e in ('post', 'dequeue') and (not k or k[1] not in ours):
            continue
        if e == 'tracing_mark_write':
            if 'FENCE signalled' in rest:
                ev.setdefault('fence_seen', (t - t0) * 1e6)
            continue
        if e == 'pull':
            continue
        ev.setdefault(e, (t - t0) * 1e6)
    order = sorted(ev.items(), key=lambda x: x[1])
    race = ev.get('post', 1e9) < ev.get('inithw_ret', -1)
    line = f'{path}: ' + '  '.join(f'{e} {us:.0f}' for e, us in order) + \
        f'   => post {"BEFORE" if race else "after"} reset return ({ev.get("post", 0) - ev.get("inithw_ret", 0):+.0f} us)'
    if 'fence_seen' in ev:
        fr = ev['fence_seen'] < ev.get('inithw_ret', -1)
        line += f'; fence seen {"BEFORE" if fr else "after"} reset return ({ev["fence_seen"] - ev.get("inithw_ret", 0):+.0f} us)'
    print(line)
