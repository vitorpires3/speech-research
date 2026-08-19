#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, html, json, math, re, unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VOWELS = ("i", "e", "a", "o", "u")
MISSING = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing", "prefer not to say"}


def args_parser():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="QAP/MRQAP social analysis of KDE overlap matrices")
    p.add_argument("--overlap-csv", type=Path, default=root/"results/kde_3d_by_vowel_speaker/kde_pairwise_overlap.csv")
    p.add_argument("--metadata-csv", type=Path, default=root/"data/raw/social_metadata.csv")
    p.add_argument("--output-dir", type=Path, default=root/"results/qap_social_kde_overlap")
    p.add_argument("--id-column", default="")
    p.add_argument("--gender-column", default="")
    p.add_argument("--nationality-column", default="")
    p.add_argument("--income-column", default="")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--min-group-size", type=int, default=2)
    a = p.parse_args()
    if a.permutations < 99: p.error("--permutations must be at least 99")
    return a


def read_csv(path):
    last = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try: return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e: last = e
    raise RuntimeError(f"Could not read {path}: {last}")


def plain(x):
    if pd.isna(x): return ""
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(c for c in s if not unicodedata.combining(c)).strip().lower()
    return re.sub(r"\s+", " ", s)


def canon(x):
    s = plain(x).replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(csv|wav|mp3|flac|xml|textgrid|txt)$", "", s)
    s = re.sub(r"[_\s]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def variants(x):
    s = canon(x); out = {s} if s else set()
    m = re.match(r"^([0-9a-f]{8})(?:-|$)", s)
    if m: out.add(m.group(1))
    return out


def seed_for(text, seed):
    h = hashlib.sha256(text.encode()).digest()
    return (seed + int.from_bytes(h[:4], "little")) % (2**32-1)


def bh(p):
    p = pd.to_numeric(p, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan); idx = np.flatnonzero(np.isfinite(p))
    if not len(idx): return out
    vals = p[idx]; order = np.argsort(vals); sv = vals[order]; m = len(sv)
    adj = np.minimum.accumulate((sv*m/np.arange(1,m+1))[::-1])[::-1]
    rev = np.empty_like(order); rev[order] = np.arange(m)
    out[idx] = np.clip(adj[rev], 0, 1)
    return out


def detect_named(df, requested, words):
    if requested:
        for c in df.columns:
            if c == requested or str(c).lower() == requested.lower(): return str(c)
        raise KeyError(f"Column {requested!r} not found. Available: {list(df.columns)}")
    names = {str(c): re.sub(r"[^a-z0-9]+", "_", plain(c)).strip("_") for c in df.columns}
    for c,n in names.items():
        if n in words: return c
    for c,n in names.items():
        if any(w in n for w in words): return c
    return None


def detect_id(meta, acoustic_ids, requested=""):
    if requested:
        col = detect_named(meta, requested, set())
        return col, pd.DataFrame([{"column":col,"match_rate":score_id(meta,col,acoustic_ids),"selected":True}])
    rows=[]
    for c in meta.columns:
        rate=score_id(meta,str(c),acoustic_ids)
        rows.append({"column":str(c),"match_rate":rate,"selected":False})
    d=pd.DataFrame(rows).sort_values("match_rate",ascending=False).reset_index(drop=True)
    if d.empty or d.loc[0,"match_rate"] < .5:
        raise RuntimeError("Could not detect the metadata ID column (best match below 50%). Rerun with --id-column COLUMN.")
    col=str(d.loc[0,"column"]); d.loc[d["column"]==col,"selected"]=True
    return col,d


def score_id(meta, col, ids):
    index={}
    for i,v in meta[col].items():
        for k in variants(v): index.setdefault(k,set()).add(i)
    ok=0
    for a in ids:
        hits=set()
        for k in variants(a): hits |= index.get(k,set())
        ok += len(hits)==1
    return ok/len(ids) if ids else 0


def match_metadata(meta, id_col, ids):
    index={}
    for i,v in meta[id_col].items():
        for k in variants(v): index.setdefault(k,set()).add(i)
    rows=[]; audit=[]
    for a in ids:
        hits=set()
        for k in variants(a): hits |= index.get(k,set())
        if len(hits)==1:
            i=next(iter(hits)); r=meta.loc[i].copy(); r["acoustic_id"]=a; rows.append(r)
            audit.append({"acoustic_id":a,"status":"matched","metadata_row":i,"metadata_id":meta.loc[i,id_col]})
        else:
            audit.append({"acoustic_id":a,"status":"unmatched" if not hits else "ambiguous","metadata_row":"|".join(map(str,sorted(hits))),"metadata_id":""})
    matched=pd.DataFrame(rows)
    if matched.empty: matched=pd.DataFrame(columns=["acoustic_id"])
    return matched.set_index("acoustic_id",drop=False), pd.DataFrame(audit)


def clean_cat(s):
    x=s.map(plain); return x.where(~x.isin(MISSING),np.nan)


def income_order(s):
    num=pd.to_numeric(s,errors="coerce")
    if num.notna().sum() >= max(5,int(.7*s.notna().sum())): return num.astype(float),"numeric"
    c=clean_cat(s); mp={}
    keys={"low":1,"lower":1,"poor":1,"faible":1,"bas":1,"baixo":1,"baixa":1,
          "middle":2,"medium":2,"mid":2,"moyen":2,"medio":2,"media":2,
          "high":3,"upper":3,"rich":3,"eleve":3,"alto":3,"alta":3}
    for v in c.dropna().unique():
        hit=[score for word,score in keys.items() if word in str(v)]
        if hit: mp[str(v)]=float(np.median(hit))
    if len(mp)>=2: return c.map(mp).astype(float),"ordered labels"
    mp={}
    for v in c.dropna().unique():
        nums=[float(z.replace(",",".")) for z in re.findall(r"\d+(?:[.,]\d+)?",str(v))]
        if nums: mp[str(v)]=float(np.mean(nums[:2]))
    if len(mp)>=2: return c.map(mp).astype(float),"numeric ranges"
    return pd.Series(np.nan,index=s.index,dtype=float),"unavailable"


def social_table(matched, cols, min_n):
    out=pd.DataFrame(index=matched.index); counts=[]; warnings=[]; income_method="unavailable"
    for role in ("gender","nationality","income"):
        c=cols.get(role)
        if not c: warnings.append(f"No {role} column detected."); continue
        x=clean_cat(matched[c]); vc=x.value_counts(dropna=True)
        out[role]=x.where(x.isin(vc[vc>=min_n].index),np.nan)
        for g,n in vc.items(): counts.append({"variable":role,"source_column":c,"category":g,"n_speakers":int(n),"retained":bool(n>=min_n)})
    if cols.get("income"):
        out["income_order"],income_method=income_order(matched[cols["income"]])
        if out["income_order"].nunique(dropna=True)<2:
            out["income_order"]=np.nan; warnings.append("Income could not be ordered automatically; same-category tests are still available.")
    return out,pd.DataFrame(counts),warnings,income_method


def validate_overlap(df):
    need={"vowel","speaker_1","speaker_2","overlap_coefficient"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"Overlap CSV missing columns: {sorted(miss)}")
    d=df.copy(); d["vowel"]=d["vowel"].astype(str).str.strip().str.lower()
    d["speaker_1"]=d["speaker_1"].astype(str).str.strip(); d["speaker_2"]=d["speaker_2"].astype(str).str.strip()
    d["overlap_coefficient"]=pd.to_numeric(d["overlap_coefficient"],errors="coerce").clip(0,1)
    return d[np.isfinite(d["overlap_coefficient"])].copy()


def pair_matrix(d,speakers):
    arr=np.full((len(speakers),len(speakers)),np.nan,dtype=float)
    np.fill_diagonal(arr,1.0)
    m=pd.DataFrame(arr,index=speakers,columns=speakers)
    for r in d.itertuples(index=False):
        if r.speaker_1 in m.index and r.speaker_2 in m.index:
            m.loc[r.speaker_1,r.speaker_2]=r.overlap_coefficient; m.loc[r.speaker_2,r.speaker_1]=r.overlap_coefficient
    return m


def acoustic_matrices(d):
    speakers=sorted(set(d.speaker_1)|set(d.speaker_2)); out={}
    for v in VOWELS:
        x=d[d.vowel==v]
        if not x.empty: out[f"/{v}/"]=pair_matrix(x,speakers)
    def add(names,label):
        arr=[out[n].to_numpy(float) for n in names if n in out]
        if len(arr)<2:return
        stack=np.stack(arr); a=np.nanmean(stack,axis=0); a[np.sum(np.isfinite(stack),axis=0)<len(arr)]=np.nan; np.fill_diagonal(a,1)
        out[label]=pd.DataFrame(a,index=speakers,columns=speakers)
    add([f"/{v}/" for v in VOWELS],"All vowels")
    add([f"/{v}/" for v in VOWELS if v!="u"],"All except /u/")
    return out


def upper(a): return a[np.triu_indices(a.shape[0],1)]


def corr(x,y,spearman=False):
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<3:return np.nan
    if spearman:
        x=pd.Series(x).rank().to_numpy(); y=pd.Series(y).rank().to_numpy()
    x=x-x.mean(); y=y-y.mean(); den=math.sqrt(float((x*x).sum()*(y*y).sum()))
    return float((x*y).sum()/den) if den else np.nan


def perm_p(obs,null,direction="positive"):
    z=null[np.isfinite(null)]
    two=(1+np.sum(np.abs(z)>=abs(obs)))/(len(z)+1)
    one=(1+np.sum(z>=obs))/(len(z)+1) if direction=="positive" else (1+np.sum(z<=obs))/(len(z)+1)
    return float(two),float(one)


def qap_same(m,cats,nperm,seed):
    sp=[s for s in m.index if s in cats.index and pd.notna(cats.loc[s])]
    a=m.loc[sp,sp].to_numpy(float); g=cats.loc[sp].astype(str).to_numpy(); same=(g[:,None]==g[None,:]).astype(float)
    av=upper(a); sv=upper(same); ok=np.isfinite(av); av=av[ok]; sv=sv[ok]
    within=av[sv==1]; between=av[sv==0]
    if not len(within) or not len(between): return None
    obs=corr(av,sv); rng=np.random.default_rng(seed); null=np.empty(nperm)
    for k in range(nperm):
        q=g[rng.permutation(len(g))]; null[k]=corr(av,upper((q[:,None]==q[None,:]).astype(float))[ok])
    p2,p1=perm_p(obs,null,"positive")
    return dict(n_speakers=len(sp),n_pairs=len(av),n_within_pairs=len(within),n_between_pairs=len(between),correlation=obs,
                mean_within_overlap=float(within.mean()),mean_between_overlap=float(between.mean()),within_minus_between=float(within.mean()-between.mean()),
                p_two_sided=p2,p_expected_direction=p1)


def qap_income(m,order,nperm,seed):
    sp=[s for s in m.index if s in order.index and np.isfinite(order.loc[s])]
    a=1-m.loc[sp,sp].to_numpy(float); x=order.loc[sp].to_numpy(float); sd=np.abs(x[:,None]-x[None,:])
    av=upper(a); sv=upper(sd); ok=np.isfinite(av)&np.isfinite(sv); av=av[ok]; sv=sv[ok]
    if len(av)<3 or np.std(sv)==0:return None
    obs=corr(av,sv,True); rng=np.random.default_rng(seed); null=np.empty(nperm)
    for k in range(nperm):
        q=x[rng.permutation(len(x))]; null[k]=corr(av,upper(np.abs(q[:,None]-q[None,:]))[ok],True)
    p2,p1=perm_p(obs,null,"positive")
    return dict(n_speakers=len(sp),n_pairs=len(av),n_within_pairs=np.nan,n_between_pairs=np.nan,correlation=obs,
                mean_within_overlap=np.nan,mean_between_overlap=np.nan,within_minus_between=np.nan,p_two_sided=p2,p_expected_direction=p1)


def run_qap(mats,social,nperm,seed):
    rows=[]
    for name,m in mats.items():
        for col,label in (("gender","Same gender"),("nationality","Same nationality"),("income","Same income category")):
            if col in social:
                r=qap_same(m,social[col],nperm,seed_for(name+col,seed))
                if r: rows.append({"matrix":name,"social_variable":label,"test_type":"Same-category Pearson QAP","expected_direction":"positive",**r})
        if "income_order" in social and social.income_order.nunique(dropna=True)>=2:
            r=qap_income(m,social.income_order,nperm,seed_for(name+"income_distance",seed))
            if r: rows.append({"matrix":name,"social_variable":"Income distance","test_type":"Ordinal-distance Spearman QAP","expected_direction":"positive",**r})
    d=pd.DataFrame(rows)
    if not d.empty:
        d["q_global_two_sided"]=bh(d.p_two_sided); d["q_global_expected_direction"]=bh(d.p_expected_direction)
        d["q_within_variable"]=d.groupby("social_variable")["p_two_sided"].transform(lambda x:bh(x))
        d=d.sort_values(["q_global_two_sided","p_two_sided"],na_position="last")
    return d


def ols(y,X):
    inv=np.linalg.pinv(X.T@X); b=inv@X.T@y; fit=X@b; res=y-fit; df=max(1,len(y)-np.linalg.matrix_rank(X))
    se=np.sqrt(np.clip(np.diag(inv)*(res@res/df),0,np.inf)); t=np.divide(b,se,out=np.full_like(b,np.nan),where=se>0)
    sst=((y-y.mean())**2).sum(); r2=1-(res@res)/sst if sst else np.nan
    return b,se,t,r2


def mrqap_one(m,social,nperm,seed):
    needed=[c for c in ("gender","nationality") if c in social]
    if "income_order" in social and social.income_order.nunique(dropna=True)>=2: needed.append("income_order")
    elif "income" in social: needed.append("income")
    sp=[s for s in m.index if s in social.index and (not needed or social.loc[s,needed].notna().all())]
    if len(sp)<8:return pd.DataFrame(),{"n_speakers":len(sp),"status":"insufficient complete speakers"}
    Y=m.loc[sp,sp].to_numpy(float); names=[]; mats=[]
    if "gender" in social: x=social.loc[sp,"gender"].astype(str).to_numpy(); names.append("same_gender"); mats.append((x[:,None]==x[None,:]).astype(float))
    if "nationality" in social: x=social.loc[sp,"nationality"].astype(str).to_numpy(); names.append("same_nationality"); mats.append((x[:,None]==x[None,:]).astype(float))
    if "income_order" in social and social.loc[sp,"income_order"].nunique()>=2:
        x=social.loc[sp,"income_order"].to_numpy(float); z=np.abs(x[:,None]-x[None,:]); z/=z.max(); names.append("income_distance"); mats.append(z)
    elif "income" in social:
        x=social.loc[sp,"income"].astype(str).to_numpy(); names.append("same_income_category"); mats.append((x[:,None]==x[None,:]).astype(float))
    if not names:return pd.DataFrame(),{"n_speakers":len(sp),"status":"no predictors"}
    idx=np.triu_indices(len(sp),1); y=Y[idx]; xv=[z[idx] for z in mats]; ok=np.isfinite(y)
    for z in xv:ok&=np.isfinite(z)
    if not ok.all(): return pd.DataFrame(),{"n_speakers":len(sp),"status":"incomplete overlap matrix"}
    X=np.column_stack([np.ones(len(y)),*xv]); b,se,t,r2=ols(y,X); rng=np.random.default_rng(seed); null=np.empty((nperm,len(names))); rnull=np.empty(nperm)
    for k in range(nperm):
        p=rng.permutation(len(sp)); yp=Y[np.ix_(p,p)][idx]; _,_,tp,rp=ols(yp,X); null[k]=tp[1:]; rnull[k]=rp
    rows=[]
    for j,n in enumerate(names):
        direction="negative" if n=="income_distance" else "positive"; p2,p1=perm_p(float(t[j+1]),null[:,j],direction)
        rows.append({"predictor":n,"coefficient":float(b[j+1]),"standard_error_ols":float(se[j+1]),"t_statistic":float(t[j+1]),
                     "p_two_sided_permutation":p2,"p_expected_direction":p1,"expected_sign":direction})
    global_p=(1+np.sum(rnull>=r2))/(nperm+1)
    return pd.DataFrame(rows),{"n_speakers":len(sp),"n_pairs":len(y),"predictors":", ".join(names),"r_squared":float(r2),"global_p":float(global_p),"status":"OK"}


def run_mrqap(mats,social,nperm,seed):
    frames=[]; sums=[]
    for name,m in mats.items():
        d,s=mrqap_one(m,social,nperm,seed_for(name+"mrqap",seed)); sums.append({"matrix":name,**s})
        if not d.empty:d.insert(0,"matrix",name);frames.append(d)
    out=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["q_global_two_sided"]=bh(out.p_two_sided_permutation); out["q_within_predictor"]=out.groupby("predictor")["p_two_sided_permutation"].transform(lambda x:bh(x))
        out=out.sort_values(["q_global_two_sided","p_two_sided_permutation"])
    return out,pd.DataFrame(sums)


def table(df):
    d=df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]): d[c]=d[c].map(lambda x:"NA" if not np.isfinite(x) else f"{x:.4f}")
    return d.to_html(index=False,border=0,escape=True)


def plots(qap,mrqap,assets):
    def bar(df,label_col,value_col,title,xlabel,path):
        if df.empty:return
        lab=df[label_col].astype(str); val=pd.to_numeric(df[value_col],errors="coerce").to_numpy(); order=np.argsort(np.nan_to_num(val,nan=-np.inf))
        fig,ax=plt.subplots(figsize=(11,max(6,.32*len(df))),constrained_layout=True); y=np.arange(len(df)); ax.barh(y,val[order]); ax.axvline(0,lw=1)
        ax.set_yticks(y);ax.set_yticklabels(lab.iloc[order],fontsize=8);ax.set_xlabel(xlabel);ax.set_title(title);ax.grid(axis="x",alpha=.25);fig.savefig(path,dpi=220,bbox_inches="tight");plt.close(fig)
    if not qap.empty:
        x=qap.copy();x["label"]=x.social_variable+" — "+x.matrix
        bar(x,"label","correlation","QAP acoustic–social associations","Observed QAP correlation",assets/"qap_correlations.png")
        y=x[x.test_type.str.startswith("Same-category")]
        bar(y,"label","within_minus_between","Within-group minus between-group KDE overlap","Mean overlap difference",assets/"within_between.png")
    if not mrqap.empty:
        x=mrqap.copy();x["label"]=x.predictor+" — "+x.matrix
        bar(x,"label","coefficient","Exploratory MRQAP coefficients","Coefficient predicting KDE overlap",assets/"mrqap_coefficients.png")


def report(path,a,id_col,cols,income_method,qap,mrqap,msum,audit,counts,candidates,warnings):
    summary=[]
    if not qap.empty:
        best=qap.loc[qap.correlation.abs().idxmax()]; summary.append(f"Largest bivariate association: <b>{html.escape(str(best.social_variable))}</b>, {html.escape(str(best.matrix))}, r={best.correlation:.3f}, p={best.p_two_sided:.4f}, q={best.q_global_two_sided:.4f}.")
        summary.append(f"{int((qap.q_global_two_sided<.05).sum())} bivariate result(s) survived global FDR q&lt;.05.")
    if not mrqap.empty: summary.append(f"{int((mrqap.q_global_two_sided<.05).sum())} adjusted MRQAP coefficient(s) survived global FDR q&lt;.05.")
    src=pd.DataFrame([{"role":"ID","column":id_col},{"role":"Gender","column":cols.get("gender")},{"role":"Nationality","column":cols.get("nationality")},{"role":"Income","column":cols.get("income")},{"role":"Income ordering","column":income_method}])
    warn="".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>No automatic warnings.</li>"
    css="""body{font-family:Arial;margin:2rem;line-height:1.45;color:#222}h2{margin-top:2.7rem;border-bottom:2px solid #ddd}.note{background:#f5f5f5;border-left:4px solid #888;padding:1rem;max-width:1150px}.tbl{overflow-x:auto;border:1px solid #ddd;margin-bottom:2rem}table{border-collapse:collapse;width:100%;font-size:.8rem}th,td{border:1px solid #ddd;padding:.45rem;text-align:right;white-space:nowrap}th{background:#eee}td:first-child,th:first-child{text-align:left}img{max-width:100%;border:1px solid #ddd;margin:1rem 0 2rem}code{background:#f2f2f2}"""
    body=f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>QAP social analysis</title><style>{css}</style></head><body>
<h1>QAP social analysis of speaker-level 3D KDE overlap</h1>
<p>This report tests whether socially similar speaker pairs also have more similar KDE density distributions.</p>
<div class='note'><b>Main interpretation:</b> positive same-category correlations mean higher overlap within a social category. Positive income-distance correlations mean larger income differences accompany larger acoustic dissimilarities. Permutation p-values account for dependence among speaker pairs.</div>
<h2>Automatic summary</h2>{''.join('<p>'+x+'</p>' for x in summary)}
<h2>Inputs</h2><p><code>{html.escape(str(a.overlap_csv.resolve()))}</code><br><code>{html.escape(str(a.metadata_csv.resolve()))}</code><br>{a.permutations:,} permutations; seed {a.seed}.</p><div class='tbl'>{table(src)}</div>
<h2>QAP correlations</h2><img src='assets/qap_correlations.png'><img src='assets/within_between.png'><div class='tbl'>{table(qap) if not qap.empty else '<p>No valid QAP result.</p>'}</div>
<h2>Exploratory MRQAP regression</h2><p>The model predicts pairwise KDE overlap from all available social matrices simultaneously. Its p-values use node-label matrix permutations. Ordinary OLS standard errors are descriptive only.</p><img src='assets/mrqap_coefficients.png'><div class='tbl'>{table(mrqap) if not mrqap.empty else '<p>No valid MRQAP result.</p>'}</div>
<h3>Model summaries</h3><div class='tbl'>{table(msum)}</div>
<h2>Metadata category counts</h2><div class='tbl'>{table(counts)}</div>
<h2>Identifier matching audit</h2><div class='tbl'>{table(audit)}</div><h3>ID-column candidates</h3><div class='tbl'>{table(candidates)}</div>
<h2>Warnings and cautions</h2><ul>{warn}<li>A detectable association does not imply separated clusters or causality.</li><li>With 43 speakers, group balance matters more than the nominal number of pairs.</li><li>MRQAP is exploratory, especially when social predictors are correlated.</li></ul></body></html>"""
    path.write_text(body,encoding="utf-8")


def main():
    a=args_parser(); out=a.output_dir.resolve(); assets=out/"assets"; assets.mkdir(parents=True,exist_ok=True)
    print("=== QAP social analysis ===");print("Overlap:",a.overlap_csv.resolve());print("Metadata:",a.metadata_csv.resolve());print("Permutations:",f"{a.permutations:,}")
    ov=validate_overlap(read_csv(a.overlap_csv.resolve())); meta=read_csv(a.metadata_csv.resolve()); ids=sorted(set(ov.speaker_1)|set(ov.speaker_2))
    id_col,candidates=detect_id(meta,ids,a.id_column); matched,audit=match_metadata(meta,id_col,ids); n=(audit.status=="matched").sum()
    candidates.to_csv(out/"metadata_id_column_candidates.csv",index=False);audit.to_csv(out/"metadata_match_audit.csv",index=False)
    print(f"Matched IDs: {n}/{len(ids)} using column {id_col}")
    if n<8: raise RuntimeError("Fewer than 8 IDs matched. Inspect metadata_match_audit.csv and rerun with --id-column COLUMN.")
    cols={"gender":detect_named(matched,a.gender_column,{"gender","sex","genre","sexo","sexe"}),
          "nationality":detect_named(matched,a.nationality_column,{"nationality","nationalite","country","country_of_origin","origin_country","pais","nation","citizenship"}),
          "income":detect_named(matched,a.income_column,{"income","income_group","income_band","household_income","family_income","revenue","salary","salaire","renda","faixa_de_renda","revenu"})}
    social,counts,warnings,income_method=social_table(matched,cols,a.min_group_size);counts.to_csv(out/"metadata_group_counts.csv",index=False)
    print("Detected columns:",cols,"income order:",income_method)
    if not any(c in social for c in ("gender","nationality","income","income_order")): raise RuntimeError("No usable social variable found. Use explicit --gender-column/--nationality-column/--income-column.")
    mats=acoustic_matrices(ov); qap=run_qap(mats,social,a.permutations,a.seed); qap.to_csv(out/"qap_correlation_results.csv",index=False);print("QAP complete")
    mrqap,msum=run_mrqap(mats,social,a.permutations,a.seed);mrqap.to_csv(out/"mrqap_results.csv",index=False);msum.to_csv(out/"mrqap_model_summary.csv",index=False);print("MRQAP complete")
    plots(qap,mrqap,assets)
    for f in ("qap_correlations.png","within_between.png","mrqap_coefficients.png"):
        p=assets/f
        if not p.exists():
            fig,ax=plt.subplots(figsize=(8,3));ax.text(.5,.5,"No valid result available",ha="center",va="center");ax.axis("off");fig.savefig(p,dpi=180,bbox_inches="tight");plt.close(fig)
    if n<len(ids):warnings.append(f"Only {n}/{len(ids)} acoustic IDs matched uniquely.")
    (out/"analysis_warnings.txt").write_text("\n".join(warnings) if warnings else "No warnings.",encoding="utf-8")
    (out/"qap_analysis_config.json").write_text(json.dumps({"overlap_csv":str(a.overlap_csv.resolve()),"metadata_csv":str(a.metadata_csv.resolve()),"id_column":id_col,"social_columns":cols,"income_order_method":income_method,"permutations":a.permutations,"seed":a.seed,"matched_ids":int(n),"total_ids":len(ids)},indent=2,ensure_ascii=False),encoding="utf-8")
    report(out/"qap_social_report.html",a,id_col,cols,income_method,qap,mrqap,msum,audit,counts,candidates,warnings)
    print("\nDone:",out/"qap_social_report.html")

if __name__=="__main__": main()
