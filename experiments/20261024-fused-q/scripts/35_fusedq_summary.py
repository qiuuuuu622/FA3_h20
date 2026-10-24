"""Summarize results/int8bf_fusedq_bench.jsonl (34_int8bf_fusedq_bench.py --timing)."""
import json,pathlib,sys
rows=[json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines()];N=len(rows)
keys=['fa3_int8bf_n96','fa3_int8bf_fusedq_n96','fa3_int8bf_fusedq_n80','ctrl_bf16','bf16','fp8_promote1_c128','q_prep']
for k in keys:
    e=[r['errors_vs_bf16'][k] for r in rows if k in r['errors_vs_bf16']]
    s=[r['errors_vs_sim'][k] for r in rows if k in r['errors_vs_sim']]
    m=[r['medians_ms'][k] for r in rows]
    line=f"{k:24s} mean_ms={sum(m)/N:.4f} speedup_vs_bf16={sum(r['speedup_vs_bf16'][k] for r in rows)/N:.4f}"
    if e:line+=(f" rel_bf16={sum(x['relative_l2'] for x in e)/len(e)*100:.4f}% worst={max(x['relative_l2'] for x in e)*100:.4f}%"
                f" strict_pass={sum(x['close'] for x in e)}/{len(e)} elem_frac={sum(x['element_pass_fraction'] for x in e)/len(e):.4f}"
                f" finite={sum(x['finite'] for x in e)}/{len(e)}")
    if s:line+=f" rel_sim={sum(x['relative_l2'] for x in s)/len(s)*100:.4f}%"
    if k in rows[0]['repeat_bitwise']:line+=f" bitwise_repeat={sum(r['repeat_bitwise'][k] for r in rows)}/{N}"
    print(line)
# fused vs prequantized+unfused q_prep (end-to-end Q path), per sample
for smp in sorted({r['capture'][:9] for r in rows}):
    rs=[r for r in rows if r['capture'].startswith(smp)]
    f=lambda k:sum(r['medians_ms'][k] for r in rs)/len(rs)
    print(smp,f"bf16={f('bf16'):.3f} pre_n96+qprep={f('fa3_int8bf_n96')+f('q_prep'):.3f} fusedq_n96={f('fa3_int8bf_fusedq_n96'):.3f}"
          f" fusedq_n80={f('fa3_int8bf_fusedq_n80'):.3f} speedup_fusedq_n96={f('bf16')/f('fa3_int8bf_fusedq_n96'):.4f}")
