import json, subprocess, sys
TASKS=["arm_pose_nymeria","full_body_pose_nymeria","odometry_nymeria_local","microgesture_openwatch",
       "language_humanml3d","pd_pads","har_hhar","har_mhealth","har_prism","har_userwatch"]
B=sys.argv[1] if len(sys.argv)>1 else "abl_norot"
print(f"ref (full model) MINUS {B}  — positive = the ablated arm is BETTER than the full model")
print(f"{'task':<24}{'diff':>9}{'95% CI':>22}  verdict")
for t in TASKS:
    try:
        out=subprocess.run([sys.executable,"-m","bench.bootstrap","--pair",t,"corpus_ref__probe__s42",f"{B}__probe__s42","--no-pairs"],
                           capture_output=True,text=True,timeout=900).stdout
        d=json.loads(out[out.index("{"):out.rindex("}")+1])
    except Exception as e:
        print(f"{t:<24}  -- ({type(e).__name__})"); continue
    hib=d["higher_is_better"]; diff=d["diff"]; lo,hi=d["bca_lo"],d["bca_hi"]
    # sign so that positive always means "ablated arm is better"
    s = -1 if hib else 1
    v, l, h = s*diff, min(s*lo,s*hi), max(s*lo,s*hi)
    verdict = ("ablated BETTER" if v>0 else "ablated worse") if d["excludes_zero"] else "no difference"
    print(f"{t:<24}{v:>+9.4f}  [{l:>+8.4f},{h:>+8.4f}]  {verdict}  (n={d['n_groups']} {d['resample_level']}s)")
