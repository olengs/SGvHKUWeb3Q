from roostoo import Roostoo
import os, json, re, textwrap
import dotenv
dotenv.load_dotenv()
from polymarket import PolyMarket

MODE = os.getenv("MODE")
API_KEY = os.getenv(f"{MODE}_API_KEY")
SECRET_KEY = os.getenv(f"{MODE}_API_SECRET")

DAILY_SLUGS = [
    ("Mar 17", "bitcoin-price-on-march-17"),
    ("Mar 18", "bitcoin-price-on-march-18"),
    ("Mar 19", "bitcoin-price-on-march-19"),
    ("Mar 20", "bitcoin-price-on-march-20"),
    ("Mar 21", "bitcoin-price-on-march-21"),
    ("Mar 22", "bitcoin-price-on-march-22"),
    ("Mar 23", "bitcoin-price-on-march-23"),
]

def extract_bracket_label(question: str) -> str:
    q = question.lower()
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
  BTC / USD — <b>90-day price history</b> &amp; <b>Polymarket bracket odds · Mar 17–23 2026</b>
  &nbsp;·&nbsp; Generated: <b>{generated}</b>
</div>

<div class="row">
  <div class="panel">
    <div class="plbl">Historical price — <b>90 days · spot ${spot:,}</b></div>
    <div class="wrap" style="height:460px;"><canvas id="cHist"></canvas></div>
  </div>
  <div class="panel">
    <div class="plbl">Polymarket odds — <b>all days side by side · green=high prob · red=low · Mar 21 not listed</b></div>
    <div class="wrap" style="height:460px;"><canvas id="cFore"></canvas></div>
  </div>
</div>

<div class="foot">
  Source: CoinGecko (history) · Polymarket CLOB API (live odds) · <b>{generated}</b>
</div>
<button onclick="dl()">&#8595; Download PNG</button>

<script>
const SPOT = {spot};

// Live Polymarket CLOB odds — fetched {generated}
const LIVE_ODDS = {live_odds_json};

const DAYS = Object.keys(LIVE_ODDS);

const BRACKET_ORDER = [
  "<62k","<64k","62-64k","64-66k","66-68k","68-70k",
  "70-72k","72-74k","74-76k","76-78k","78-80k","80-82k",">80k",">82k"
];

function allBrackets() {{
  const seen = new Set();
  for (const day of DAYS)
    for (const row of LIVE_ODDS[day]) seen.add(row.b);
  return BRACKET_ORDER.filter(b => seen.has(b));
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
    const end = Math.floor(new Date("2026-03-17T23:59:59Z").getTime()/1000);
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
    const base=new Date("2026-03-17");
    let p=96500;
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
    data:{{
      labels,
      datasets:[
        {{data:prices,borderColor:'#7a7570',borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}},
        {{data:prices.map((p,i)=>i===prices.length-1?p:null),
          borderColor:'#1D9E75',backgroundColor:'#1D9E75',
          pointRadius:prices.map((_,i)=>i===prices.length-1?6:0),
          showLine:false,fill:false}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        tooltip:{{
          backgroundColor:'rgba(8,10,14,0.95)',titleColor:'#ccc8c0',bodyColor:'#666',
          titleFont:{{family:"'Courier New',monospace",size:10}},
          bodyFont:{{family:"'Courier New',monospace",size:10}},
          callbacks:{{label:c=>c.raw===null?null:' $'+c.raw.toLocaleString()}},
          filter:i=>i.raw!==null
        }}
      }},
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
    return {{
      label: day,
      data: vals,
      backgroundColor: vals.map(p=>pColor(p,mx,0.80)),
      borderColor:     vals.map(p=>pColor(p,mx,1.0)),
      borderWidth:0.5,
      borderRadius:2,
    }};
  }});

  new Chart(document.getElementById('cFore'), {{
    type:'bar',
    data:{{labels:brackets, datasets}},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{display:true,position:'bottom',
          labels:{{color:'#555',font:{{size:9,family:"'Courier New',monospace"}},boxWidth:10,padding:8}}}},
        tooltip:{{
          backgroundColor:'rgba(8,10,14,0.95)',titleColor:'#ccc8c0',bodyColor:'#888',
          titleFont:{{family:"'Courier New',monospace",size:10}},
          bodyFont:{{family:"'Courier New',monospace",size:10}},
          callbacks:{{
            title:items=>`Bracket: ${{items[0].label}}`,
            label:c=>` ${{c.dataset.label}}: ${{(c.raw*100).toFixed(1)}}%`
          }}
        }}
      }},
      scales:{{
        x:{{ticks:{{color:'#555',font:{{size:9,family:"'Courier New',monospace"}},maxRotation:30}},
           grid:{{display:false}},border:{{color:'#232730'}}}},
        y:{{min:0,
           ticks:{{color:'#3a3f4a',font:{{size:9,family:"'Courier New',monospace"}},
                  callback:v=>(v*100).toFixed(0)+'%'}},
           grid:{{color:'rgba(255,255,255,0.025)'}},border:{{color:'#232730'}}}}
      }}
    }}
  }});
}}

function dl() {{
  const c1=document.getElementById('cHist');
  const c2=document.getElementById('cFore');
  const pad=20,top=44,bot=32;
  const W=c1.width+c2.width+pad*3;
  const H=Math.max(c1.height,c2.height)+top+bot;
  const out=document.createElement('canvas');
  out.width=W; out.height=H;
  const ctx=out.getContext('2d');
  ctx.fillStyle='#0f1117'; ctx.fillRect(0,0,W,H);
  ctx.font="11px 'Courier New'"; ctx.fillStyle='#555';
  ctx.fillText('BTC/USD  ·  90-day history + Polymarket bracket odds  ·  Mar 17–23 2026  ·  Generated {generated}', pad, 26);
  ctx.fillText('Spot: $'+SPOT.toLocaleString(), pad, H-10);
  ctx.drawImage(c1,pad,top);
  ctx.drawImage(c2,c1.width+pad*2,top);
  const a=document.createElement('a');
  a.download='btc-polymarket-forecast.png';
  a.href=out.toDataURL('image/png');
  a.click();
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


def build_html(bands: dict, spot: int, generated: str) -> str:
    return HTML_TEMPLATE.format(
        spot=spot,
        generated=generated,
        live_odds_json=json.dumps(bands, indent=2),
    )


if __name__ == "__main__":
    import datetime

    broker = Roostoo(API_KEY, SECRET_KEY)
    market = PolyMarket()

    all_results = {}
    bands = {}

    for label, slug in DAILY_SLUGS:
        print(f"\n{'='*50}")
        print(f"  Bitcoin Price — {label}  [{slug}]")
        print(f"{'='*50}")
        try:
            data = market.get_market_data(slug)
            if not data:
                print(f"  [!] No liquid markets found (slug may not exist yet or spreads too wide)")
                all_results[label] = []
                continue
            PolyMarket.print_market_data(data)
            all_results[label] = data
            bands[label] = [
                {
                    "b":   extract_bracket_label(item["Question"]),
                    "p":   round(item["p"], 4),
                    "bid": item["Yes"]["bid"],
                    "ask": item["Yes"]["ask"],
                }
                for item in sorted(data, key=lambda x: x["Question"])
            ]
        except Exception as e:
            print(f"  [!] Error fetching {label}: {e}")
            all_results[label] = []

    # ── summary ──
    print(f"\n{'='*50}")
    print("  Top bracket per day")
    print(f"{'='*50}")
    for label, data in all_results.items():
        if not data:
            print(f"  {label}: no data")
            continue
        top = max(data, key=lambda x: x["p"])
        lbl = extract_bracket_label(top["Question"])
        print(f"  {label}: {lbl}  @  {float(top['p'])*100:.1f}%"
              f"  |  bid={top['Yes']['bid']:.3f}  ask={top['Yes']['ask']:.3f}")

    # ── write HTML ──
    SPOT = 74855  # update this to current spot when running
    generated = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    html = build_html(bands, SPOT, generated)

    output_path = "btc_viz.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  [✓] {output_path} written — open in your browser to view the chart")
    print(f"  [✓] polymarket_bands.json also available for reference")

    # also write the raw JSON for reference
    with open("polymarket_bands.json", "w") as f:
        json.dump(bands, f, indent=2)