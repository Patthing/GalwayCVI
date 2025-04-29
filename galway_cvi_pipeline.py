"""
Galway Cultural Value Index (CVI) – Unified Pipeline + Streamlit UI
===================================================================
File: galway_cvi_pipeline.py   •   Version 0.3 (28 Apr 2025)

One file, three roles:

1. Library   – functions `compute_indices()` etc.
2. CLI       – `python galway_cvi_pipeline.py --out file.csv`
3. Streamlit – `streamlit run galway_cvi_pipeline.py` for a GUI.

Edit only the INDICATORS list (table codes, weights) if you need tweaks.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# Standard libs
# ------------------------------------------------------------------
import argparse, datetime as dt, json, logging, os, pathlib, re, sys
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd, requests

BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

FIRST_YEAR = 2015
CURRENT_YEAR = dt.datetime.now().year - 1  # most sources are t‑1

# ---------------------- Indicator catalogue ------------------------
def _default_weight(dim: str) -> float:
    return {
        "cultural_vibrancy": 1 / 3,
        "creative_economy": 1 / 4,
        "enabling_env": 1 / 4,
    }[dim]

@dataclass
class Indicator:
    name: str
    provider: str              # pxstat | screen_ireland | ticketmaster | file
    source_id: str             # table, endpoint, or filename
    selector: Dict[str, str | list[str]] | None
    transform: str             # "sum", "count", "sum / population * 1000" etc.
    dimension: str
    weight: float = field(default_factory=lambda: 0.0)
    postprocess: str | None = None
    notes: str = ""

    def __post_init__(self):
        if not self.weight:
            self.weight = _default_weight(self.dimension)

INDICATORS: List[Indicator] = [
   # Indicator(
#     name="creative_industry_employment",
#     provider="pxstat",
#     source_id="BRA/BRA30",
#     selector={
#         "Statistic": "Number of Enterprises",
#         "County": "Galway",
#         "NACE Rev 2": ["J58", "J59", "J60", "M71", "R90", "R91"],
#     },
#     transform="sum",
#     dimension="creative_economy",
#     postprocess="per_capita",
# ),
    # Indicator(
#     name="film_production_days",
#     provider="screen_ireland",
#     source_id="https://api.screenireland.ie/productions?county=Galway",
#     selector=None,
#     transform="sum",
#     dimension="creative_economy",
#     weight=0.25,
# ),
    Indicator(
        name="ticketmaster_culture_events",
        provider="ticketmaster",
        source_id="dmaId=602&classificationName=Arts%20%26%20Theatre",
        selector=None,
        transform="count",
        dimension="cultural_vibrancy",
        weight=0.34,
    ),
]

# ------------------------ Provider helpers -------------------------
API_PXSTAT = "https://ws.cso.ie/public/api.restful/PxStat/JSON/statistics/{}"
TICKETMASTER_DISCOVERY = (
    "https://app.ticketmaster.com/discovery/v2/events.json?apikey={api_key}&{query}"
    "&size=200&startDateTime={year}-01-01T00:00:00Z&endDateTime={year}-12-31T23:59:59Z"
)

def _raw(label: str, year: int | None = None) -> pathlib.Path:
    suf = f"_{year}" if year else ""
    safe = re.sub(r"[^\w-]", "_", label)
    return RAW_DIR / f"{safe}{suf}.json"

def fetch_pxstat(table:str) -> pd.DataFrame:
    p=_raw(table)
    if p.exists():
        txt=p.read_text()
    else:
        url=API_PXSTAT.format(table)
        txt=requests.get(url,timeout=60).text
        p.write_text(txt)
    data=json.loads(txt)["statistic"]
    return pd.json_normalize(data,max_level=1)

def fetch_screen_ireland(url:str)->pd.DataFrame:
    p=_raw("screen_ireland")
    if p.exists():
        txt=p.read_text()
    else:
        txt=requests.get(url,timeout=60).text
        p.write_text(txt)
    return pd.DataFrame(json.loads(txt))

def fetch_ticketmaster(query:str,year:int)->int:
    key=os.getenv("TICKETMASTER_API_KEY")
    if not key:
        raise RuntimeError("Set env var TICKETMASTER_API_KEY")
    url=TICKETMASTER_DISCOVERY.format(api_key=key,query=query,year=year)
    p=_raw("ticketmaster",year)
    if p.exists():
        txt=p.read_text()
    else:
        txt=requests.get(url,timeout=60).text
        p.write_text(txt)
    return int(json.loads(txt).get("page",{}).get("totalElements",0))

def fetch_file(path:str)->pd.DataFrame:
    fp=pathlib.Path(path)
    return pd.read_csv(fp) if fp.suffix==".csv" else pd.read_json(fp)

# ---------------------- Helpers & transforms -----------------------
def _filter(df:pd.DataFrame,sel): 
    if not sel: return df
    for k,v in sel.items():
        df=df[df[k].isin(v)] if isinstance(v,list) else df[df[k]==v]
    return df

def _fill(s:pd.Series)->pd.Series: return s.fillna(method="ffill")

def _population(y:int)->int:
    pops={2015:258058,2016:258552,2017:259003,2018:260997,2019:263073,
          2020:264640,2021:266943,2022:270160,2023:272400,2024:274650}
    return pops.get(y,pops[max(pops)])

def _eval(expr:str,local:Dict): 
    return eval(expr,{"__builtins__":{}},local)

# -------------------- Build indicator series -----------------------
def series_for(ind:Indicator)->pd.Series:
    if ind.provider=="pxstat":
        df=fetch_pxstat(ind.source_id)
        df=_filter(df,ind.selector)[["Year","value"]]
        df=df.rename(columns={"value":ind.name}).set_index("Year")
    elif ind.provider=="screen_ireland":
        df=fetch_screen_ireland(ind.source_id).rename(
            columns={"year":"Year","production_days":ind.name}).set_index("Year")
    elif ind.provider=="ticketmaster":
        df=pd.Series({y:fetch_ticketmaster(ind.source_id,y) for y in range(FIRST_YEAR,CURRENT_YEAR+1)},name=ind.name).to_frame()
    elif ind.provider=="file":
        df=fetch_file(ind.source_id).set_index("Year")[[ind.name]]
    else:
        raise ValueError(ind.provider)

    for y in range(FIRST_YEAR,CURRENT_YEAR+1):
        if y not in df.index: df.loc[y]=pd.NA
    df.sort_index(inplace=True)
    df[ind.name]=_fill(df[ind.name].astype(float))

    if ind.transform not in {"sum","count"}:
        if ind.transform.startswith(("sum /","count /")):
            df[ind.name]=[_eval(ind.transform,{"sum":v,"count":v,"population":_population(y)}) for y,v in df[ind.name].items()]
        else:
            raise ValueError(ind.transform)

    if ind.postprocess=="per_capita":
        df[ind.name]=[v/_population(y)*1000 for y,v in df[ind.name].items()]
    return df[ind.name]

# ------------------------- Aggregate CVI ---------------------------
def _z(s:pd.Series)->pd.Series: return (s-s.mean())/s.std(ddof=0)

def compute_indices()->pd.DataFrame:
    mat=pd.concat([series_for(i) for i in INDICATORS],axis=1)
    zmat=mat.apply(_z,axis=0)
    dims={}
    for i in INDICATORS:
        dims.setdefault(i.dimension,0)
        dims[i.dimension]+=zmat[i.name]*i.weight
    df=pd.DataFrame(dims)
    df["CVI"]=df.apply(lambda r:(r+1).prod()**(1/len(r))-1,axis=1)
    return df

# ----------------------------- CLI ---------------------------------
def _cli(out:pathlib.Path|None):
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    res=compute_indices()
    res.index.name="Year"
    out=out or PROC_DIR/"galway_cvi_series.csv"
    res.to_csv(out)
    logging.info("Saved → %s",out)

# ------------------------- Streamlit UI ----------------------------
def _run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Galway CVI",layout="wide")
    st.title("Galway Cultural Value Index")

    st.sidebar.header("Options")
    yr=st.sidebar.slider("Year range",FIRST_YEAR,CURRENT_YEAR,(FIRST_YEAR,CURRENT_YEAR))
    subs=["cultural_vibrancy","creative_economy","enabling_env","CVI"]
    show=st.sidebar.multiselect("Sub-indices",subs,default=["CVI"])

    if st.button("Run / Refresh"):
        with st.spinner("Crunching data…"):
            df=compute_indices().loc[yr[0]:yr[1]]
        st.success("Done")
        st.line_chart(df[show])
        st.dataframe(df.round(3))
        st.download_button("Download CSV",df.to_csv().encode(),"galway_cvi_series.csv","text/csv")

    st.caption("v0.3 • Data cached locally to minimise API calls.")

if __name__=="__main__":
    if any("streamlit" in a for a in sys.argv):
        _run_streamlit()
    else:
        p=argparse.ArgumentParser(description="Galway CVI builder")
        p.add_argument("--out",type=pathlib.Path,help="CSV output path")
        _cli(p.parse_args().out)
