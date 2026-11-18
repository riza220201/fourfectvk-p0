#!/usr/bin/env python3
"""p0sum.py <p0run trace>: the P0 timeline of a vkhang run around the loss -- the probe's markers, our context's
kbase events (post/dequeue), job completions on the hardware (kbase_jd_done of our atoms), the reset if any, and
every kbase_gpu_munmap of our context (a BO taken off the GPU). Times in ms relative to the first marker that
mentions vkWaitForFences/DEVICE_LOST. Checks: no munmap of ours between the loss and the settle barrier's
hardware completion (the barrier = our last atom to complete on the GPU before the first munmap)."""
import re, sys
LINE = re.compile(r'^\s*(?P<comm>.+?)-(?P<pid>\d+)\s+\[(?P<cpu>\d+)\]\s+\S+\s+(?P<t>\d+\.\d+): (?P<ev>[a-z_]+): (?P<rest>.*)$')
rows = [(float(m['t']), m['comm'].strip(), int(m['pid']), m['ev'], m['rest'])
        for m in (LINE.match(l) for l in open(sys.argv[1], errors='replace')) if m]
probe = {r[2] for r in rows if r[3] == 'tracing_mark_write' and r[4].startswith('faultprobe: ')}
probe_tgid_comm = {r[1] for r in rows if r[2] in probe}
ours = set()
for r in rows:
    if r[3] in ('dequeue', 'gpumunmap', 'destroy') and (r[2] in probe or r[1] in probe_tgid_comm):
        k = re.search(r'kctx=(0x[0-9a-f]+)', r[4])
        if k and r[3] == 'dequeue':
            ours.add(k[1])
anchor = [r[0] for r in rows if r[3] == 'tracing_mark_write' and ('vkWaitForFences' in r[4] or 'EXPORTED FENCE' in r[4])]
t0 = anchor[0] if anchor else rows[0][0]
posts = set()
for t, c, p, e, rest in rows:
    k = re.search(r'kctx=(0x[0-9a-f]+)', rest)
    if e == 'post' and k and k[1] in ours:
        posts.add(re.search(r'katom=(0x[0-9a-f]+)', rest)[1])
out = []
for t, c, p, e, rest in rows:
    k = re.search(r'kctx=(0x[0-9a-f]+)', rest)
    a = re.search(r'katom=(0x[0-9a-f]+)', rest)
    rel = (t - t0) * 1e3
    if rel < -6000 or rel > 4000:
        continue
    if e == 'tracing_mark_write' and rest.startswith('faultprobe: '):
        out.append((rel, '--', rest[12:100]))
    elif e in ('post', 'dequeue', 'gpumunmap', 'destroy', 'zap') and k and k[1] in ours:
        out.append((rel, e, f'{c}-{p} {a[1] if a else ""}'))
    elif e == 'jddone' and a and a[1] in posts:
        out.append((rel, 'jddone', a[1]))
    elif e in ('resetworker', 'backendreset', 'hardstop'):
        out.append((rel, e, ''))
for rel, e, x in out:
    if e != 'gpumunmap' or rel > -50:
        print(f'{rel:10.3f} ms  {e:10s} {x}')
mun = [rel for rel, e, x in out if e == 'gpumunmap' and rel > 0]
print(f'our kctx {sorted(ours)}; munmaps after the loss: {len(mun)}, first at {mun[0]:.3f} ms' if mun else 'no munmap after the loss')
