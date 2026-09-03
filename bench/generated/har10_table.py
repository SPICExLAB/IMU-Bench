#!/usr/bin/env python3
"""Full 10-dataset HAR table: every encoder x every HAR dataset, macro-F1."""
import json, os, sys
DS=["wisdm","utd_mhad","pamap2","recofit","hhar","mhealth","samosa","shoaib","ut_watch","prism"]
GROUPS=[("Inertia-1 (650,000 h NHANES, acc-only)",
         [("inertia1_art","probe","ART-S probe"),("inertia1_art","finetune","ART-S finetune"),
          ("inertia1_patchtst","probe","PatchTST-S probe"),("inertia1_patchtst","finetune","PatchTST-S finetune")]),
        ("Board encoders (leaked pool, 300k steps)",
         [("mae_11src","probe","MAE-11src probe"),("mae_11src","finetune","MAE-11src finetune"),
          ("mae_5src","probe","MAE-5src probe"),("mae_noaltattn","probe","MAE-noAA probe"),
          ("mae_acconly","probe","MAE-acc probe"),("simmtm","probe","SimMTM probe"),
          ("focal","probe","FOCAL probe"),("relcon","probe","RelCon probe"),
          ("scratch","scratch","Scratch")]),
        ("Leak-free 2x2 arms (525 / 4,325 h, 50k steps)",
         [("mae_nhanes2x2_a","probe","A 200h z-score"),("mae_nhanes2x2_b","probe","B 4000h z-score"),
          ("mae_nhanes2x2_c","probe","C 200h physical"),("mae_nhanes2x2_d","probe","D 4000h physical")]),
        ("Recipe ablation (514 h, 30k steps)",
         [("mae_recipe_r0","probe","R0 baseline"),("mae_recipe_r1","probe","R1 phys+yaw+lr"),
          ("mae_recipe_r2","probe","R2 +band loss"),("mae_recipe_r3","probe","R3 +modality mask")])]
def get(e,p,d):
    f=f"bench/results/har_{d}/{e}__{p}__s42.json"
    if not os.path.exists(f): f=f"bench/results/har_{d}/{e}__{p}__s0.json"
    return json.load(open(f))["metrics"]["macro_f1"] if os.path.exists(f) else None
W=max(len(l) for _,rows in GROUPS for _,_,l in rows)+1
hdr=" "*W+"".join(d[:8].rjust(9) for d in DS)+"    mean7    mean10"
for title,rows in GROUPS:
    print(f"\n=== {title} ===\n{hdr}")
    for e,p,lab in rows:
        v=[get(e,p,d) for d in DS]
        s7=[x for x in v[:7] if x is not None]; s10=[x for x in v if x is not None]
        m7=("%.3f"%(sum(s7)/7)) if len(s7)==7 else f"({len(s7)}/7)"
        m10=("%.3f"%(sum(s10)/10)) if len(s10)==10 else f"({len(s10)}/10)"
        print(lab.ljust(W)+"".join((("%.3f"%x) if x is not None else "   ·  ").rjust(9) for x in v)+f"{m7:>9}{m10:>10}")
