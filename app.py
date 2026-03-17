"""
app.py — BTC Polymarket live fetch + Monte Carlo calibration
============================================================
  1. Auto-generates DAILY_SLUGS for today + next 7 days
  2. Fetches live Polymarket CLOB odds for each day
  3. Feeds live odds directly into btc_monte_carlo_normal.py calibrator
  4. Writes btc_viz.html (chart) + polymarket_bands.json + MC report
"""

import os, json, re, datetime
import dotenv
dotenv.load_dotenv()

from polymarket import PolyMarket
from roostoo import Roostoo

MODE       = os.getenv("MODE")
API_KEY    = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

# ─────────────────────────────────────────────────────────────
# 1. AUTO-GENERATE SLUGS: today → today + 7 days
# ─────────────────────────────────────────────────────────────

def generate_daily_slugs(num_days: int = 7) -> list[tuple[str, str]]:
    """
    Returns list of (label, slug) for today through today + num_days-1.
    e.g. ("Mar 17", "bitcoin-price-on-march-17")
    """
    today = datetime.date.today()
    slugs = []

    for i in range(num_days):
        d = today + datetime.timedelta(days=i)

        # Cross-platform formatting (works on Windows/macOS/Linux)
        label = f"{d.day} {d.strftime('%b')}"      # "17 Mar"
        month = d.strftime("%B").lower()           # "march"
        day = str(d.day)                           # "17"
        slug = f"bitcoin-price-on-{month}-{day}"
        short = f"{d.strftime('%b')} {d.day}"      # "Mar 17"

        slugs.append((short, slug))

    return slugs

DAILY_SLUGS = generate_daily_slugs(num_days=7)

# ─────────────────────────────────────────────────────────────
# 2. BRACKET LABEL EXTRACTOR
# ─────────────────────────────────────────────────────────────

def extract_bracket_label(question: str) -> str:
    q    = question.lower()
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", q)
            if int(n.replace(",", "")) >= 1000]
    if not nums:
        return question
    if "greater than" in q or "above" in q:
        return f">${nums[0]//1000}k"
    if "less than" in q or "below" in q or "dip" in q:
        return f"<{nums[0]//1000}k"
    if len(nums) >= 2:
        lo, hi = sorted(nums[:2])
        return f"{lo//1000}-{hi//1000}k"
    return f"{nums[0]//1000}k"


def parse_bracket_bounds(question: str) -> tuple[float | None, float | None]:
    """Extract (lo, hi) numeric bounds from a question string."""
    q    = question.lower()
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", q)
            if int(n.replace(",", "")) >= 1000]
    if not nums:
        return None, None
    if "greater than" in q or "above" in q:
        return float(nums[0]), None
    if "less than" in q or "below" in q or "dip" in q:
        return None, float(nums[0])
    if len(nums) >= 2:
        lo, hi = sorted(nums[:2])
        return float(lo), float(hi)
    return None, None

# ─────────────────────────────────────────────────────────────
# 3. FETCH LIVE POLYMARKET DATA
# ─────────────────────────────────────────────────────────────

def fetch_all_days(market: PolyMarket) -> dict:
    """
    Returns dict:
        {
          "Mar 17": {
              "bands": [{"b": "72-74k", "p": 0.425, "bid": 0.42, "ask": 0.43}, ...],
              "brackets": [("72-74k", 72000, 74000, 0.425), ...]   # for MC calibrator
          },
          ...
        }
    """
    results = {}

    print(f"\n[FETCH] Today is {datetime.date.today()}  —  fetching {len(DAILY_SLUGS)} days\n")

    for label, slug in DAILY_SLUGS:
        print(f"{'='*52}")
        print(f"  {label}  [{slug}]")
        print(f"{'='*52}")

        try:
            data = market.get_market_data(slug)
        except Exception as e:
            print(f"  [!] Error: {e}")
            results[label] = None
            continue

        if not data:
            print(f"  [!] No liquid markets (slug may not exist yet or spreads too wide)")
            results[label] = None
            continue

        PolyMarket.print_market_data(data)

        bands    = []
        brackets = []

        for item in sorted(data, key=lambda x: x["Question"]):
            lbl      = extract_bracket_label(item["Question"])
            lo, hi   = parse_bracket_bounds(item["Question"])
            p        = round(item["p"], 4)
            bid      = item["Yes"]["bid"]
            ask      = item["Yes"]["ask"]

            bands.append({"b": lbl, "p": p, "bid": bid, "ask": ask})
            brackets.append((lbl, lo, hi, p))

        results[label] = {"bands": bands, "brackets": brackets}

    return results

# ─────────────────────────────────────────────────────────────
# 4. HTML TEMPLATE
# ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BTC Polymarket Forecast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0f1117;color:#ddd8d0;font-family:'Courier New',monospace;padding:28px;}}
  .title{{font-size:11px;color:#555;letter-spacing:.1em;text-transform:uppercase;margin-bottom:20px;}}
  .title b{{color:#aaa;font-weight:normal;}}
  .row{{display:grid;grid-template-columns:1fr 1.65fr;gap:16px;}}
  .panel{{background:#161921;border:1px solid #232730;border-radius:5px;padding:14px 16px;}}
  .plbl{{font-size:9px;color:#444;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;}}
  .plbl b{{color:#666;font-weight:normal;}}
  .wrap{{position:relative;width:100%;}}
  .foot{{margin-top:10px;font-size:9px;color:#363b46;letter-spacing:.05em;}}
  .foot b{{color:#4a5060;font-weight:normal;}}
  button{{margin-top:12px;background:transparent;border:1px solid #1D9E75;color:#1D9E75;
          font-family:'Courier New',monospace;font-size:10px;padding:6px 14px;border-radius:3px;
          cursor:pointer;letter-spacing:.06em;text-transform:uppercase;}}
  button:hover{{background:#1D9E75;color:#0f1117;}}
</style>
</head>
<body>

<div class="title">
  BTC / USD — <b>90-day price history</b> &amp;
  <b>Polymarket bracket odds · {date_range}</b>
  &nbsp;·&nbsp; Generated: <b>{generated}</b>
</div>

<div class="row">
  <div class="panel">
    <div class="plbl">Historical price — <b>90 days · spot ${spot:,}</b></div>
    <div class="wrap" style="height:460px;"><canvas id="cHist"></canvas></div>
  </div>
  <div class="panel">
    <div class="plbl">Polymarket odds — <b>all days side by side · green=high prob · red=low</b></div>
    <div class="wrap" style="height:460px;"><canvas id="cFore"></canvas></div>
  </div>
</div>

<div class="foot">
  Source: CoinGecko (history) · Polymarket CLOB API (live odds) · <b>{generated}</b>
</div>
<button onclick="dl()">&#8595; Download PNG</button>

<script>
const SPOT = {spot};
const LIVE_ODDS = {live_odds_json};
const DAYS = Object.keys(LIVE_ODDS);

const BRACKET_ORDER = [
  "<60k","<62k","<64k","<66k",
  "60-62k","62-64k","64-66k","66-68k","68-70k",
  "70-72k","72-74k","74-76k","76-78k","78-80k","80-82k","82-84k","84-86k","86-88k",
  ">80k",">82k",">84k",">86k",">88k",">90k"
];

function allBrackets() {{
  const seen = new Set();
  for (const day of DAYS) for (const row of LIVE_ODDS[day]) seen.add(row.b);
  const ordered = BRACKET_ORDER.filter(b => seen.has(b));
  // append any unseen brackets not in BRACKET_ORDER
  for (const day of DAYS) for (const row of LIVE_ODDS[day])
    if (!ordered.includes(row.b)) ordered.push(row.b);
  return ordered;
}}

function pColor(p, mx, a) {{
  const t = Math.min(p / mx, 1);
  let r, g, b;
  if (t > 0.6) {{
    const s=(t-0.6)/0.4;
    r=Math.round(29+(16-29)*s); g=Math.round(158+(200-158)*s); b=Math.round(117+(80-117)*s);
  }} else if (t > 0.3) {{
    const s=(t-0.3)/0.3;
    r=Math.round(186+(29-186)*s); g=Math.round(117+(158-117)*s); b=Math.round(23+(117-23)*s);
  }} else {{
    const s=t/0.3;
    r=Math.round(175+(186-175)*s); g=Math.round(35+(117-35)*s); b=Math.round(35+(23-35)*s);
  }}
  return `rgba(${{r}},${{g}},${{b}},${{a||1}})`;
}}

async function fetchHist() {{
  try {{
    const end = Math.floor(Date.now()/1000);
    const res = await fetch(
      `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from=${{end-91*86400}}&to=${{end}}`,
      {{signal:AbortSignal.timeout(10000)}});
    const d = await res.json();
    if (!d.prices || d.prices.length < 20) throw 0;
    const step = Math.max(1, Math.floor(d.prices.length/90));
    const prices=[], labels=[];
    for (let i=0;i<d.prices.length;i+=step) {{
      const [ts,px]=d.prices[i];
      labels.push(new Date(ts).toLocaleDateString('en-GB',{{month:'short',day:'numeric'}}));
      prices.push(Math.round(px));
    }}
    prices[prices.length-1]=SPOT;
    return {{prices,labels}};
  }} catch {{
    const prices=[], labels=[];
    const base=new Date();
    let p=SPOT*1.05;
    for (let i=89;i>=0;i--) {{
      const d=new Date(base); d.setDate(d.getDate()-i);
      labels.push(d.toLocaleDateString('en-GB',{{month:'short',day:'numeric'}}));
      if (i===0) prices.push(SPOT);
      else {{ p=p*Math.exp(-0.00035+0.030*(Math.random()*2-1)); prices.push(Math.round(p)); }}
    }}
    return {{prices,labels}};
  }}
}}

function buildHist(prices, labels) {{
  new Chart(document.getElementById('cHist'), {{
    type:'line',
    data:{{labels,datasets:[
      {{data:prices,borderColor:'#7a7570',borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}},
      {{data:prices.map((p,i)=>i===prices.length-1?p:null),
        borderColor:'#1D9E75',backgroundColor:'#1D9E75',
        pointRadius:prices.map((_,i)=>i===prices.length-1?6:0),showLine:false,fill:false}}
    ]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},tooltip:{{
        backgroundColor:'rgba(8,10,14,0.95)',titleColor:'#ccc8c0',bodyColor:'#666',
        titleFont:{{family:"'Courier New',monospace",size:10}},
        bodyFont:{{family:"'Courier New',monospace",size:10}},
        callbacks:{{label:c=>c.raw===null?null:' $'+c.raw.toLocaleString()}},
        filter:i=>i.raw!==null
      }}}},
      scales:{{
        x:{{ticks:{{color:'#3a3f4a',font:{{size:9,family:"'Courier New',monospace"}},maxTicksLimit:10,maxRotation:30}},
           grid:{{color:'rgba(255,255,255,0.025)'}},border:{{color:'#232730'}}}},
        y:{{ticks:{{color:'#3a3f4a',font:{{size:9,family:"'Courier New',monospace"}},callback:v=>'$'+(v/1000).toFixed(0)+'k'}},
           grid:{{color:'rgba(255,255,255,0.025)'}},border:{{color:'#232730'}}}}
      }}
    }}
  }});
}}

function buildFore() {{
  const brackets = allBrackets();
  const allP = DAYS.flatMap(day => {{
    const map = Object.fromEntries(LIVE_ODDS[day].map(r=>[r.b,r.p]));
    return brackets.map(b=>map[b]||0);
  }});
  const mx = Math.max(...allP);
  const datasets = DAYS.map(day => {{
    const map = Object.fromEntries(LIVE_ODDS[day].map(r=>[r.b,r.p]));
    const vals = brackets.map(b=>map[b]||0);
    return {{label:day,data:vals,
      backgroundColor:vals.map(p=>pColor(p,mx,0.80)),
      borderColor:vals.map(p=>pColor(p,mx,1.0)),
      borderWidth:0.5,borderRadius:2}};
  }});
  new Chart(document.getElementById('cFore'), {{
    type:'bar',data:{{labels:brackets,datasets}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:true,position:'bottom',
          labels:{{color:'#555',font:{{size:9,family:"'Courier New',monospace"}},boxWidth:10,padding:8}}}},
        tooltip:{{backgroundColor:'rgba(8,10,14,0.95)',titleColor:'#ccc8c0',bodyColor:'#888',
          titleFont:{{family:"'Courier New',monospace",size:10}},
          bodyFont:{{family:"'Courier New',monospace",size:10}},
          callbacks:{{title:items=>`Bracket: ${{items[0].label}}`,
            label:c=>` ${{c.dataset.label}}: ${{(c.raw*100).toFixed(1)}}%`}}
        }}
      }},
      scales:{{
        x:{{ticks:{{color:'#555',font:{{size:9,family:"'Courier New',monospace"}},maxRotation:30}},
           grid:{{display:false}},border:{{color:'#232730'}}}},
        y:{{min:0,ticks:{{color:'#3a3f4a',font:{{size:9,family:"'Courier New',monospace"}},
          callback:v=>(v*100).toFixed(0)+'%'}},
          grid:{{color:'rgba(255,255,255,0.025)'}},border:{{color:'#232730'}}}}
      }}
    }}
  }});
}}

function dl() {{
  const c1=document.getElementById('cHist'), c2=document.getElementById('cFore');
  const pad=20,top=44,bot=32;
  const W=c1.width+c2.width+pad*3, H=Math.max(c1.height,c2.height)+top+bot;
  const out=document.createElement('canvas'); out.width=W; out.height=H;
  const ctx=out.getContext('2d');
  ctx.fillStyle='#0f1117'; ctx.fillRect(0,0,W,H);
  ctx.font="11px 'Courier New'"; ctx.fillStyle='#555';
  ctx.fillText('BTC/USD  ·  90-day history + Polymarket bracket odds  ·  Generated {generated}',pad,26);
  ctx.fillText('Spot: $'+SPOT.toLocaleString(),pad,H-10);
  ctx.drawImage(c1,pad,top); ctx.drawImage(c2,c1.width+pad*2,top);
  const a=document.createElement('a');
  a.download='btc-polymarket-forecast.png'; a.href=out.toDataURL('image/png'); a.click();
}}

(async()=>{{
  const {{prices,labels}} = await fetchHist();
  buildHist(prices,labels);
  buildFore();
}})();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# 5. MONTE CARLO CALIBRATION (inline, no import needed)
# ─────────────────────────────────────────────────────────────

def run_mc_calibration(S0: float, live_data: dict) -> dict:
    """
    For each day that has live bracket data, fit Normal(mu_r, sigma_r)
    to match Polymarket odds using KL-divergence minimisation.
    Returns dict of day -> {mu_r, sigma_r, kl, grade}
    """
    import numpy as np
    from scipy.stats import norm
    from scipy.optimize import minimize

    def bracket_probs(S0, brackets, mu_r, sigma_r):
        probs = np.zeros(len(brackets))
        for i, (_, lo, hi, _) in enumerate(brackets):
            lo_r = np.log(lo/S0) if lo is not None else -np.inf
            hi_r = np.log(hi/S0) if hi is not None else  np.inf
            cdf_hi = 1.0 if hi_r ==  np.inf else norm.cdf(hi_r, mu_r, sigma_r)
            cdf_lo = 0.0 if lo_r == -np.inf else norm.cdf(lo_r, mu_r, sigma_r)
            probs[i] = max(cdf_hi - cdf_lo, 0.0)
        return probs / max(probs.sum(), 1e-9)

    def kl(p, q, eps=1e-9):
        p = np.clip(p, eps, 1); p /= p.sum()
        q = np.clip(q, eps, 1); q /= q.sum()
        return float(np.sum(p * np.log(p/q)))

    mc_results = {}

    for day, day_data in live_data.items():
        if day_data is None:
            print(f"[MC] {day}: skipped (no data)")
            continue

        brackets = day_data["brackets"]
        raw      = np.array([b[3] for b in brackets])
        target   = raw / raw.sum()

        def obj(params):
            mu_r, log_s = params
            q = bracket_probs(S0, brackets, mu_r, np.exp(log_s))
            return kl(target, q)

        # Grid warm-start
        best_kl, best_x = np.inf, None
        for mu_r in np.linspace(-0.10, 0.05, 35):
            for log_s in np.linspace(np.log(0.005), np.log(0.12), 35):
                q = bracket_probs(S0, brackets, mu_r, np.exp(log_s))
                k = kl(target, q)
                if k < best_kl:
                    best_kl = k
                    best_x  = (mu_r, log_s)

        res     = minimize(obj, best_x, method="L-BFGS-B",
                           options={"maxiter":10000,"ftol":1e-15,"gtol":1e-12})
        mu_r    = res.x[0]
        sigma_r = np.exp(res.x[1])
        kl_val  = res.fun
        grade   = ("EXCELLENT" if kl_val < 0.005 else
                   "GOOD"      if kl_val < 0.020 else
                   "FAIR"      if kl_val < 0.100 else "POOR")

        model_p = bracket_probs(S0, brackets, mu_r, sigma_r)

        # Print bracket comparison
        W = 80
        print(f"\n{'='*W}")
        print(f"  MC CALIBRATION — {day}  KL={kl_val:.8f}  [{grade}]")
        print(f"  {'Bracket':>8}  {'Polymarket':>11}  {'Model':>10}  {'Diff':>8}  {'Pass':>5}")
        print(f"  {'-'*(W-2)}")
        for i, (lbl, _, _, _) in enumerate(brackets):
            pm, md = target[i], model_p[i]
            diff   = md - pm
            ok     = "✓" if abs(diff)<=0.030 else ("~" if abs(diff)<=0.060 else "✗")
            print(f"  {lbl:>8}  {pm:>10.2%}  {md:>10.2%}  {diff:>+7.2%}  {ok:>5}")
        ann_vol = sigma_r * (365**0.5) * 100
        print(f"  {'─'*(W-2)}")
        print(f"  mu_r={mu_r*100:+.4f}%  sigma_r={sigma_r*100:.4f}%  "
              f"ann_vol={ann_vol:.1f}%  [{grade}]")
        print(f"{'='*W}")

        mc_results[day] = {
            "mu_r":    mu_r,
            "sigma_r": sigma_r,
            "kl":      kl_val,
            "grade":   grade,
        }

    return mc_results


# ─────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    broker = Roostoo(API_KEY, SECRET_KEY)
    market = PolyMarket()

    generated  = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    today      = datetime.date.today()
    date_range = f"{DAILY_SLUGS[0][0]} – {DAILY_SLUGS[-1][0]} {today.year}"

    # ── Step 1: fetch live Polymarket data ──
    live_data = fetch_all_days(market)

    # ── Step 2: summary of top bracket per day ──
    print(f"\n{'='*52}")
    print("  Top bracket per day (live CLOB)")
    print(f"{'='*52}")
    for label, day_data in live_data.items():
        if not day_data:
            print(f"  {label}: no data")
            continue
        top = max(day_data["brackets"], key=lambda x: x[3])
        print(f"  {label}: {top[0]:>8}  @ {top[3]*100:.1f}%")

    # ── Step 3: get current BTC spot price ──
    try:
        ticker_data = broker.get_ticker("BTC/USD")
        SPOT = int(float(ticker_data["Data"]["BTC/USD"]["LastPrice"]))
        print(f"\n[SPOT] Live BTC/USD from Roostoo: ${SPOT:,}")
    except Exception as e:
        # Fallback: extract from Polymarket implied median if Roostoo fails
        SPOT = 74855
        print(f"\n[SPOT] Roostoo unavailable ({e}) — using fallback ${SPOT:,}")

    # ── Step 4: run Monte Carlo calibration ──
    print(f"\n[MC] Running Normal distribution calibration against live Polymarket odds ...\n")
    mc_results = run_mc_calibration(float(SPOT), live_data)

    # ── Step 5: print MC summary ──
    print(f"\n{'='*65}")
    print("  MONTE CARLO CALIBRATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Day':<10} {'mu_r':>9} {'sigma_r':>9} {'Ann.Vol':>8} {'KL':>12} {'Grade':>10}")
    print(f"  {'-'*63}")
    for day, res in mc_results.items():
        ann = res["sigma_r"] * (365**0.5) * 100
        print(f"  {day:<10} {res['mu_r']*100:>+8.4f}% {res['sigma_r']*100:>8.4f}% "
              f"{ann:>7.1f}% {res['kl']:>12.8f} {res['grade']:>10}")
    print(f"{'='*65}")

    # ── Step 6: build HTML viz from live odds ──
    bands_for_html = {
        label: day_data["bands"]
        for label, day_data in live_data.items()
        if day_data is not None
    }

    html = HTML_TEMPLATE.format(
        spot           = SPOT,
        generated      = generated,
        date_range     = date_range,
        live_odds_json = json.dumps(bands_for_html, indent=2),
    )

    with open("btc_viz.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [✓] btc_viz.html written — open in browser")

    # ── Step 7: write JSON outputs ──
    with open("polymarket_bands.json", "w") as f:
        json.dump(bands_for_html, f, indent=2)

    mc_export = {
        day: {
            "mu_r_pct":    round(res["mu_r"]*100, 6),
            "sigma_r_pct": round(res["sigma_r"]*100, 6),
            "ann_vol_pct": round(res["sigma_r"]*(365**0.5)*100, 2),
            "kl":          round(res["kl"], 8),
            "grade":       res["grade"],
        }
        for day, res in mc_results.items()
    }
    with open("mc_calibration.json", "w") as f:
        json.dump(mc_export, f, indent=2)

    print(f"  [✓] polymarket_bands.json written")
    print(f"  [✓] mc_calibration.json written")
    print(f"\n  Run complete — {generated}")