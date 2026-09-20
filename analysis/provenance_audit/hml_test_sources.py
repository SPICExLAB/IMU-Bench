import os, csv, collections, json
R="/projects/b53017/vbc1591/HumanML3D"
idx={}
with open(f"{R}/index.csv") as f:
    for row in csv.DictReader(f):
        idx[row["new_name"].replace(".npy","")]=row
def sub(p):
    parts=p.split("/")
    return parts[2]  # ./pose_data/<subset>/...
out={}
for split in ["train","val","test"]:
    ids=[l.strip() for l in open(f"{R}/HumanML3D/{split}.txt") if l.strip()]
    aug=set(x[:-4] for x in os.listdir(f"{R}/UniMTS_data/data/aug_texts"))
    sim=set(x[:-4] for x in os.listdir(f"{R}/UniMTS_data/data/sim"))
    kept=[i for i in ids if i in aug and i in sim]
    c_all=collections.Counter(); c_kept=collections.Counter()
    miss=0
    for i in ids:
        base=i[1:] if i.startswith("M") else i
        if base not in idx: miss+=1; continue
        c_all[sub(idx[base]["source_path"])]+=1
    for i in kept:
        base=i[1:] if i.startswith("M") else i
        if base not in idx: continue
        c_kept[sub(idx[base]["source_path"])]+=1
    print(split, "ids",len(ids),"kept(aug∩sim)",len(kept),"missing_in_index",miss, "M-prefixed kept", sum(k.startswith("M") for k in kept))
    for k,v in sorted(c_all.items(), key=lambda x:-x[1]):
        print(f"   {k:28s} all={v:6d} kept={c_kept.get(k,0):6d}")
    out[split]={"all":dict(c_all),"kept":dict(c_kept),"kept_ids":kept}
json.dump(out,open("hml_split_sources.json","w"))
