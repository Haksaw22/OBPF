import json, statistics
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

with open(r"C:/Users/kulbi/Documents/Coding/OPBF/opbf2/results/e6/e6_results.json") as fh:
    d = json.load(fh)
res = d["results"]; gate = d["gate"]
print("config:", res["config"])
for k in ("opbf_transfer","opbf_scratch","sf_fair","sf_oracle"):
    v = res[k]
    print(f"{k}: n={len(v)} s2t mean={statistics.fmean(v):.1f} median={statistics.median(v)} vals={v}")
asy = res["asymptote"]
for k,v in asy.items():
    print(f"asymptote {k}: n={len(v)} mean={statistics.fmean(v):.4f} median={statistics.median(v):.4f}")
t = np.array(asy["opbf_transfer"]); sf = np.array(asy["sf_fair"])
print("transfer asymptotes:", np.round(t,3).tolist())
print("sf_fair asymptotes:", np.round(sf,3).tolist())
print("wilcoxon asymptote transfer vs sf_fair p =", wilcoxon(t, sf).pvalue)
s2t_t = np.array(res["opbf_transfer"]); s2t_sf = np.array(res["sf_fair"]); s2t_sc = np.array(res["opbf_scratch"])
print("wilcoxon s2t transfer vs sf_fair p =", wilcoxon(s2t_t, s2t_sf).pvalue)
print("mannwhitney s2t transfer<scratch p =", mannwhitneyu(s2t_t, s2t_sc, alternative="less").pvalue)
print("failures (never reach 0.9): transfer", int((s2t_t>=1600).sum()), "scratch", int((s2t_sc>=1600).sum()), "sf_fair", int((s2t_sf>=1600).sum()))
print("recovered AMI: mean=%.3f min=%.3f max=%.3f" % (statistics.fmean(res["recovered_ami"]), min(res["recovered_ami"]), max(res["recovered_ami"])))
print("AMI vals:", [round(a,3) for a in res["recovered_ami"]])
print("corr AMI vs transfer asymptote:", np.corrcoef(res["recovered_ami"], t)[0,1])
print("GATE:", json.dumps(gate, indent=1))
