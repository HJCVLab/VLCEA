import json, sys, os, glob, zipfile

def validate_one(obj):
    assert "scores" in obj and isinstance(obj["scores"], dict)
    assert "edges" in obj and isinstance(obj["edges"], list)
    # score range
    for k,v in obj["scores"].items():
        sv=float(v)
        assert 0.0<=sv<=1.0
    for e in obj["edges"]:
        assert "i" in e and "j" in e
        i=int(e["i"]); j=int(e["j"])
        assert i!=j
        if "conf" in e:
            c=float(e["conf"]); assert 0.0<=c<=1.0
        if "type" in e:
            assert e["type"] in ("hard","soft")
    return True

def main():
    if len(sys.argv)<2:
        print("Usage: python validate_constraints.py <constraints_dir_or_zip>")
        sys.exit(1)
    path=sys.argv[1]
    files=[]
    if os.path.isdir(path):
        files=glob.glob(os.path.join(path,"*.json"))
    elif path.endswith(".zip"):
        # list jsons in zip
        z=zipfile.ZipFile(path)
        files=[name for name in z.namelist() if name.endswith(".json")]
        for name in files:
            obj=json.loads(z.read(name))
            validate_one(obj)
        print(f"OK: {len(files)} JSON files in zip")
        return
    else:
        files=[path]
    ok=0
    for fp in files:
        with open(fp,"r") as f:
            obj=json.load(f)
        validate_one(obj)
        ok+=1
    print(f"OK: {ok} files validated")

if __name__=="__main__":
    main()
