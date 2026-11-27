"""Independent source provenance audit. Passive battery telemetry only."""
import base64,csv,hashlib,io,json,math,struct,urllib.request,urllib.parse
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parent
REF="67069ed00ddbebd62b71aa9bb1272415e9b15ff8"
def get(path):
    url="https://raw.githubusercontent.com/YujiaoHu/AMOVFLY-Dataset/"+REF+"/"+urllib.parse.quote(path)
    with urllib.request.urlopen(url,timeout=60) as r:
        return r.read()
def blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def parse(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
def main():
    out=[]; extracts=[]; errors=[]; metadata={}
    try:
        rows=parse(get("Flight_info.csv"))
        for row in rows: metadata[row["FlightName"]]=row
    except Exception as e: errors.append({"metadata":repr(e)})
    for line in (P/"source-manifest.txt").read_text().splitlines():
        path,want,expected=line.split("|")
        try:
            raw=get(path); actual=blob(raw)
            assert actual==want,(actual,want)
            rows=parse(raw)[:1099]
            parsed=[]
            for row in rows:
                try:
                    v=tuple(float(row[k]) for k in ("time","battery_voltage","battery_current"))
                    if all(math.isfinite(a) for a in v): parsed.append(v)
                except (ValueError,TypeError): continue
            assert len(parsed)>0
            j=0; data=bytearray(); samples=[]
            for t in range(60,156):
                while j+1<len(parsed) and abs(parsed[j+1][0]-t)<=abs(parsed[j][0]-t): j+=1
                ts,v,c=parsed[j]; assert abs(ts-t)<=.35,(t,ts)
                # JS Math.round in the original extract, including ties toward positive infinity.
                data.extend(struct.pack("<hh",math.floor(v*100+.5),math.floor(c*10+.5)))
                samples.append((ts,v,c))
            actual_extract=hashlib.sha256(data).hexdigest()
            assert actual_extract==expected,(actual_extract,expected)
            h=2166136261
            for n in data: h=((h^n)*16777619)&0xffffffff
            extracts.append("|".join([path,want,f"{h:08x}",base64.b64encode(data).decode()]))
            name=Path(path).stem
            md=metadata.get(name,{})
            assert md and md["Uav"]==name[3],("metadata mismatch",name)
            out.append(dict(path=path,source_sha=actual,extract_sha256=actual_extract,
                aircraft=md.get("Uav"),battery_label=md.get("BatteryName"),
                samples=len(samples),source_rows=len(rows),zero_current_samples=sum(x[2]==0 for x in samples),
                max_time_error=max(abs(s[0]-t) for t,s in enumerate(samples,60)),
                voltage_min=min(s[1] for s in samples),voltage_max=max(s[1] for s in samples)))
        except Exception as e: errors.append(dict(path=path,error=repr(e)))
    groups=Counter(x["aircraft"] for x in out)
    packs={}
    for x in out: packs.setdefault(x["battery_label"],set()).add(x["aircraft"])
    report=dict(upstream_commit=REF,verified=len(out),groups=dict(groups),records=out,
        shared_battery_labels={str(k):sorted(v) for k,v in packs.items() if len(v)>1},errors=errors,
        scope="12 short excerpts only; pack-label overlap is not proof of same physical pack.")
    (P/"provenance-result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    assert not errors and len(out)==12 and dict(groups)=={"Y":4,"R":4,"G":4}
    assert len({x["extract_sha256"] for x in out})==12
    (P/"reconstructed-telemetry.txt").write_text("\n".join(extracts)+"\n")
if __name__=="__main__":main()
