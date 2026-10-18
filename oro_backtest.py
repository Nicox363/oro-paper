#!/usr/bin/env python3
"""
Backtest swing semanal en el oro (y control en plata, platino y GLD).

Reglas (base):
  1. Tendencia: cierre > EMA50 y EMA20 > EMA50 -> largos;
     cierre < EMA50 y EMA20 < EMA50 -> cortos.
  2. Entrada: señal lunes-miércoles. Largo: el mínimo ha tocado la EMA20 en
     los últimos 3 días y el cierre supera el máximo del día anterior.
     Corto: espejo (el máximo toca la EMA20 y el cierre rompe el mínimo).
     Se entra a la apertura del día siguiente.
  3. Stop: 1,5 x ATR(14) contra la entrada.
  4. Take profit: 2R.
  5. Cierre obligatorio el último día de la semana (viernes).
  Máx. 1 operación por semana. Riesgo 1 % del capital de papel.

Variantes fijadas de antemano:
  A_base, B_sin_tendencia, C_sin_TP, D_lun_jue, E_2_por_semana

Uso:
  pip install yfinance pandas numpy
  python3 oro_backtest.py                  # descarga datos de Yahoo
  python3 oro_backtest.py --csv-dir datos/ # usa CSV locales (Date,Open,High,Low,Close)

Salida: informe_backtest.html y operaciones_backtest.csv
"""
import argparse
import html
import math
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

INICIO = "2000-01-01"
CORTE_OOS = "2023-10-01"          # últimos ~3 años reservados
CAPITAL = 10_000.0
RIESGO = 0.01
MC_ITER = 5000
SEED = 42

# activo: (ticker Yahoo, coste ida+vuelta en % del precio [spread + deslizamiento])
ACTIVOS = {
    "Oro (GC=F)": ("GC=F", 0.05),
    "Plata (SI=F)": ("SI=F", 0.10),
    "Platino (PL=F)": ("PL=F", 0.15),
    "GLD (ETF)": ("GLD", 0.03),
}
PRINCIPAL = "Oro (GC=F)"

# Criterios de decisión (fijados antes de ver resultados)
CRIT_EXP = 0.20
CRIT_PF = 1.30
CRIT_DD = 12.0


@dataclass(frozen=True)
class Reglas:
    nombre: str
    filtro_tendencia: bool = True
    usar_tp: bool = True
    tp_r: float = 2.0
    atr_mult: float = 1.5
    dias_senal: tuple = (0, 1, 2)      # 0=lunes ... 4=viernes
    max_por_semana: int = 1
    dias_toque: int = 3


BASE = Reglas("A_base")
VARIANTES = [
    BASE,
    replace(BASE, nombre="B_sin_tendencia", filtro_tendencia=False),
    replace(BASE, nombre="C_sin_TP", usar_tp=False),
    replace(BASE, nombre="D_lun_jue", dias_senal=(0, 1, 2, 3)),
    replace(BASE, nombre="E_2_por_semana", max_por_semana=2),
]


# ---------------------------------------------------------------- datos
def cargar_yahoo(ticker):
    import yfinance as yf
    df = yf.download(ticker, start=INICIO, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return limpiar(df)


def cargar_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return limpiar(df.set_index("Date"))


def limpiar(df):
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()
    df = df[(df["Open"] > 0) & (df["High"] >= df["Low"])]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated()].sort_index()
    return df[df.index >= INICIO]


def indicadores(df):
    d = df.copy()
    d["ema20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["dow"] = d.index.dayofweek
    iso = d.index.isocalendar()
    d["semana"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    # último día de negociación de cada semana (normalmente viernes)
    d["fin_semana"] = d["semana"] != d["semana"].shift(-1)
    return d.iloc[60:]  # calentamiento de indicadores


# ---------------------------------------------------------------- motor
def backtest(d, reglas, coste_pct, activo):
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    ema20, ema50, atr = d["ema20"].to_numpy(), d["ema50"].to_numpy(), d["atr"].to_numpy()
    dow, semana, fin = d["dow"].to_numpy(), d["semana"].to_numpy(), d["fin_semana"].to_numpy()
    fechas = d.index
    n = len(d)
    ops = []
    usadas = {}
    i = 1
    while i < n - 1:
        if dow[i] not in reglas.dias_senal or usadas.get(semana[i], 0) >= reglas.max_por_semana:
            i += 1
            continue
        ini = max(0, i - reglas.dias_toque + 1)
        if reglas.filtro_tendencia:
            alcista = c[i] > ema50[i] and ema20[i] > ema50[i]
            bajista = c[i] < ema50[i] and ema20[i] < ema50[i]
        else:
            alcista = bajista = True
        largo = alcista and np.any(l[ini:i + 1] <= ema20[ini:i + 1]) and c[i] > h[i - 1]
        corto = bajista and np.any(h[ini:i + 1] >= ema20[ini:i + 1]) and c[i] < l[i - 1]
        # la entrada (día siguiente) debe caer en la misma semana; señal ambigua -> fuera
        if largo == corto or semana[i + 1] != semana[i]:
            i += 1
            continue
        s = 1 if largo else -1

        j = i + 1
        entrada = o[j]
        riesgo = reglas.atr_mult * atr[i]
        sl = entrada - s * riesgo
        tp = entrada + s * reglas.tp_r * riesgo if reglas.usar_tp else s * math.inf
        # en coordenadas "a favor": precio*s, así largos y cortos usan la misma lógica
        sl_s, tp_s = sl * s, tp * s
        salida, motivo, k = None, None, j
        while True:
            fo = o[k] * s
            peor = (l[k] if s == 1 else h[k]) * s
            mejor = (h[k] if s == 1 else l[k]) * s
            # orden conservador: si toca SL y TP el mismo día -> SL
            if fo <= sl_s and k > j:          # gap a través del stop
                salida, motivo = o[k], "SL (gap)"
            elif peor <= sl_s:
                salida, motivo = sl, "SL"
            elif fo >= tp_s and k > j:
                salida, motivo = o[k], "TP (gap)"
            elif mejor >= tp_s:
                salida, motivo = tp, "TP"
            elif fin[k] or k == n - 1:
                salida, motivo = c[k], "Viernes"
            if salida is not None:
                break
            k += 1

        coste = entrada * coste_pct / 100
        r = (s * (salida - entrada) - coste) / riesgo
        ops.append({
            "activo": activo, "variante": reglas.nombre, "dir": "Largo" if s == 1 else "Corto",
            "fecha_senal": fechas[i].date(), "fecha_entrada": fechas[j].date(),
            "fecha_salida": fechas[k].date(), "entrada": round(entrada, 4),
            "sl": round(sl, 4), "tp": round(tp, 4) if reglas.usar_tp else None,
            "salida": round(salida, 4), "motivo": motivo, "R": round(r, 4),
        })
        usadas[semana[i]] = usadas.get(semana[i], 0) + 1
        i = k  # una posición a la vez; si salió intradía, el cierre de ese día puede dar nueva señal
    return pd.DataFrame(ops)


# ---------------------------------------------------------------- métricas
def max_dd_r(rs):
    if len(rs) == 0:
        return 0.0
    eq = np.cumsum(rs)
    return float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq, initial=0))


def metricas(ops, anios):
    if ops.empty:
        return {"n": 0}
    rs = ops["R"].to_numpy()
    gan, per = rs[rs > 0], rs[rs <= 0]
    pf = gan.sum() / abs(per.sum()) if per.sum() != 0 else math.inf
    eq = CAPITAL * np.prod(1 + RIESGO * rs)
    cagr = (eq / CAPITAL) ** (1 / anios) - 1 if anios > 0 and eq > 0 else float("nan")
    return {
        "n": len(rs), "por_anio": len(rs) / anios if anios else 0,
        "win": (rs > 0).mean() * 100, "exp": rs.mean(), "pf": pf,
        "r_gan": gan.mean() if len(gan) else 0, "r_per": per.mean() if len(per) else 0,
        "dd": max_dd_r(rs), "total_r": rs.sum(), "capital_final": eq, "cagr": cagr * 100,
        "racha_perd": racha(rs <= 0),
    }


def racha(mask):
    m = c = 0
    for x in mask:
        c = c + 1 if x else 0
        m = max(m, c)
    return m


def monte_carlo(rs, iters=MC_ITER):
    rng = np.random.default_rng(SEED)
    dds, tots = np.empty(iters), np.empty(iters)
    for t in range(iters):
        s = rng.choice(rs, size=len(rs), replace=True)
        dds[t], tots[t] = max_dd_r(s), s.sum()
    # rango esperado para 12 semanas de papel (~6 operaciones)
    n12 = 6
    doce = rng.choice(rs, size=(iters, n12), replace=True).sum(axis=1)
    return {
        "dd_p50": np.percentile(dds, 50), "dd_p95": np.percentile(dds, 95),
        "prob_negativo": (tots < 0).mean() * 100,
        "p10_12sem": np.percentile(doce, 10), "p50_12sem": np.percentile(doce, 50),
    }


def anios_entre(d):
    return (d.index[-1] - d.index[0]).days / 365.25 if len(d) > 1 else 0


# ---------------------------------------------------------------- informe
CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e4e1da;--card:#fff;--ok:#1f7a4d;--ko:#b3261e;--acc:#a8780d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151513;--fg:#ecebe6;--mut:#9b9a94;--line:#2c2b28;--card:#1d1c1a;--ok:#5cc28f;--ko:#ef7b72;--acc:#e0b04a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.mut{color:var(--mut);font-size:13px}.tw{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13.5px}th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}tr:last-child td{border-bottom:0}
.ok{color:var(--ok);font-weight:600}.ko{color:var(--ko);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mut);font-size:12.5px}
.ver{padding:14px;border-radius:8px;border:1px solid var(--line);background:var(--card);font-weight:600}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
"""


def f(x, nd=2, suf=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "∞" if isinstance(x, float) and math.isinf(x) else "—"
    return f"{x:,.{nd}f}{suf}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabla(filas, cab):
    h = "<div class=tw><table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cab) + "</tr>"
    for fila in filas:
        h += "<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>"
    return h + "</table></div>"


def fila_m(nombre, m):
    if m.get("n", 0) == 0:
        return [nombre, "0"] + ["—"] * 7
    return [html.escape(nombre), str(m["n"]), f(m["por_anio"], 1), f(m["win"], 1, " %"),
            f(m["exp"], 3, " R"), f(m["pf"]), f(m["dd"], 1, " R"), f(m["total_r"], 1, " R"),
            f(m["cagr"], 1, " %")]


CAB_M = ["", "Trades", "Trades/año", "Win %", "Expectativa", "Profit factor", "Max DD", "Total", "CAGR (1 %)"]


def curva_svg(ops_dict):
    w, hgt, pad = 900, 320, 40
    series = {k: np.cumsum(v["R"].to_numpy()) for k, v in ops_dict.items() if not v.empty}
    if not series:
        return ""
    allv = np.concatenate([np.concatenate([[0], s]) for s in series.values()])
    lo, hi = allv.min(), allv.max()
    hi = hi if hi > lo else lo + 1
    cols = ["var(--acc)", "#5b8def", "#9a6fd6", "#3fa7a0", "#d0743c"]
    out = f'<svg viewBox="0 0 {w} {hgt}" role=img aria-label="Curva de R acumulado">'
    y0 = hgt - pad - (0 - lo) / (hi - lo) * (hgt - 2 * pad)
    out += f'<line x1={pad} x2={w-10} y1={y0:.1f} y2={y0:.1f} stroke="var(--line)"/>'
    for idx, (k, s) in enumerate(series.items()):
        s = np.concatenate([[0], s])
        xs = np.linspace(pad, w - 10, len(s))
        ys = hgt - pad - (s - lo) / (hi - lo) * (hgt - 2 * pad)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        out += f'<polyline fill=none stroke="{cols[idx % 5]}" stroke-width="{2.4 if idx==0 else 1.4}" points="{pts}"/>'
        out += f'<text x={pad+6} y={18+idx*16} font-size=12 fill="{cols[idx % 5]}">{html.escape(k)}</text>'
    out += f'<text x=4 y={pad} font-size=11 fill="var(--mut)">{f(hi,0)} R</text><text x=4 y={hgt-pad} font-size=11 fill="var(--mut)">{f(lo,0)} R</text>'
    return out + "</svg>"


def informe(res, datos_info, path, anios_total):
    oro = res[PRINCIPAL]
    base_is, base_oos = oro["A_base"]["is"], oro["A_base"]["oos"]
    mis, moos, mtot = base_is["m"], base_oos["m"], oro["A_base"]["total"]["m"]
    mc = oro["A_base"]["total"]["mc"]

    checks = [
        ("Expectativa in-sample ≥ +0,20 R", mis.get("n", 0) > 0 and mis["exp"] >= CRIT_EXP),
        ("Profit factor in-sample ≥ 1,30", mis.get("n", 0) > 0 and mis["pf"] >= CRIT_PF),
        ("Max drawdown in-sample ≤ 12 R", mis.get("n", 0) > 0 and mis["dd"] <= CRIT_DD),
        ("Expectativa out-of-sample > 0", moos.get("n", 0) > 0 and moos["exp"] > 0),
    ]
    control_pos = sum(1 for a in ACTIVOS if a != PRINCIPAL and res[a]["A_base"]["total"]["m"].get("exp", -1) > 0)
    checks.append((f"Expectativa positiva en ≥ 2 de 3 activos de control ({control_pos}/3)", control_pos >= 2))
    pasa = all(ok for _, ok in checks)

    h = f"<!doctype html><html lang=es><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Backtest Oro Semanal</title><style>{CSS}</style></head><body><main>"
    h += "<h1>Backtest swing semanal · Oro</h1>"
    h += f"<p class=mut>{html.escape(datos_info)} · In-sample hasta {CORTE_OOS} · Out-of-sample desde {CORTE_OOS} · Largos y cortos · Riesgo 1 % por trade · Capital papel {f(CAPITAL,0)} €</p>"

    h += "<h2>Veredicto (regla base)</h2>"
    h += f"<div class=ver><span class={'ok' if pasa else 'ko'}>{'PASA a papel en vivo' if pasa else 'NO PASA: no se ajustan reglas para forzarlo'}</span></div><br>"
    h += tabla([[html.escape(t), f"<span class={'ok' if ok else 'ko'}>{'✔' if ok else '✘'}</span>"] for t, ok in checks], ["Criterio", ""])

    h += "<h2>Oro · regla base (2000–hoy)</h2><div class=kpis>"
    for v, lab in [(str(mtot.get('n', 0)), "operaciones"), (f(mtot.get('win', 0), 1, ' %'), "aciertos"),
                   (f(mtot.get('exp', 0), 3, ' R'), "expectativa / trade"), (f(mtot.get('pf', 0)), "profit factor"),
                   (f(mtot.get('dd', 0), 1, ' R'), "max drawdown"), (f(mtot.get('capital_final', 0), 0, ' €'), "capital final (papel)")]:
        h += f"<div class=kpi><b>{v}</b><span>{lab}</span></div>"
    h += "</div>"

    h += "<h2>Oro · in-sample vs out-of-sample</h2>"
    h += tabla([fila_m(f"{k} · IS", oro[k]["is"]["m"]) for k in oro] + [fila_m(f"{k} · OOS", oro[k]["oos"]["m"]) for k in oro], CAB_M)

    h += "<h2>Oro · curva de R acumulado por variante</h2>"
    h += curva_svg({k: oro[k]["total"]["ops"] for k in oro})

    h += "<h2>Monte Carlo (oro, regla base, 5.000 simulaciones)</h2>"
    h += tabla([
        ["Drawdown mediano", f(mc.get('dd_p50', 0), 1, ' R')],
        ["Drawdown peor 5 %", f(mc.get('dd_p95', 0), 1, ' R')],
        ["Prob. de resultado total negativo", f(mc.get('prob_negativo', 0), 1, ' %')],
        ["Papel 12 semanas (~6 trades): mediana", f(mc.get('p50_12sem', 0), 2, ' R')],
        ["Papel 12 semanas: percentil 10 (umbral de alarma)", f(mc.get('p10_12sem', 0), 2, ' R')],
    ], ["Métrica", "Valor"])

    h += "<h2>Robustez · misma regla en otros activos</h2>"
    h += tabla([fila_m(f"{a} · {k}", res[a][k]["total"]["m"]) for a in ACTIVOS for k in ("A_base",)], CAB_M)
    h += "<h2>Todos los activos y variantes (periodo completo)</h2>"
    h += tabla([fila_m(f"{a} · {k}", res[a][k]["total"]["m"]) for a in ACTIVOS for k in res[a]], CAB_M)

    ops = oro["A_base"]["total"]["ops"]
    if not ops.empty:
        ops = ops.assign(anio=pd.to_datetime(ops["fecha_entrada"]).dt.year)
        filas = []
        for y, g in ops.groupby("anio"):
            rs = g["R"].to_numpy()
            filas.append([str(y), str(len(rs)), f((rs > 0).mean() * 100, 0, " %"), f(rs.sum(), 1, " R")])
        h += "<h2>Oro · regla base por año</h2>" + tabla(filas, ["Año", "Trades", "Win %", "Total"])
        h += "<h2>Oro · largos vs cortos (regla base)</h2>" + tabla(
            [fila_m(dname, metricas(g, anios_total)) for dname, g in ops.groupby("dir")], CAB_M)
        mot = ops["motivo"].value_counts()
        h += "<h2>Oro · cómo se cierran las operaciones</h2>" + tabla(
            [[html.escape(k), str(v), f(v / len(ops) * 100, 1, " %"), f(ops.loc[ops.motivo == k, 'R'].mean(), 2, ' R')] for k, v in mot.items()],
            ["Motivo", "Trades", "%", "R medio"])

    h += "<p class=mut>Supuestos: datos diarios; si SL y TP se tocan el mismo día cuenta SL (conservador); cierre de viernes al cierre diario; futuros continuos sin ajustar por roll; costes por activo incluidos. Resultados pasados no garantizan resultados futuros.</p>"
    h += "</main></body></html>"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(h)
    return pasa


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", help="carpeta con CSV <ticker>.csv (Date,Open,High,Low,Close)")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    res, todas = {}, []
    info = []
    for nombre, (ticker, coste) in ACTIVOS.items():
        if a.csv_dir:
            p = os.path.join(a.csv_dir, f"{ticker.replace('=', '_')}.csv")
            df = cargar_csv(p)
        else:
            df = cargar_yahoo(ticker)
        if df.empty:
            raise SystemExit(f"Sin datos para {ticker}")
        d = indicadores(df)
        if nombre == PRINCIPAL:
            anios_oro = anios_entre(d)
        info.append(f"{nombre}: {d.index[0].date()}→{d.index[-1].date()}")
        corte = pd.Timestamp(CORTE_OOS)
        res[nombre] = {}
        for reglas in VARIANTES:
            ops = backtest(d, reglas, coste, nombre)
            todas.append(ops)
            if ops.empty:
                ops = pd.DataFrame(columns=["fecha_entrada", "R", "motivo"])
            fe = pd.to_datetime(ops["fecha_entrada"]) if len(ops) else pd.Series([], dtype="datetime64[ns]")
            partes = {
                "total": (ops, anios_entre(d)),
                "is": (ops[fe < corte], anios_entre(d[d.index < corte])),
                "oos": (ops[fe >= corte], anios_entre(d[d.index >= corte])),
            }
            res[nombre][reglas.nombre] = {}
            for k, (o, an) in partes.items():
                m = metricas(o, an) if len(o) else {"n": 0}
                entry = {"ops": o, "m": m}
                if k == "total" and len(o) >= 10:
                    entry["mc"] = monte_carlo(o["R"].to_numpy())
                else:
                    entry["mc"] = {}
                res[nombre][reglas.nombre][k] = entry
        print(f"✓ {nombre}: {len(d)} días")

    os.makedirs(a.out, exist_ok=True)
    pd.concat([t for t in todas if not t.empty]).to_csv(os.path.join(a.out, "operaciones_backtest.csv"), index=False)
    pasa = informe(res, " · ".join(info), os.path.join(a.out, "informe_backtest.html"), anios_oro)

    m = res[PRINCIPAL]["A_base"]["total"]["m"]
    print(f"\nOro base: {m.get('n',0)} trades | win {m.get('win',0):.1f} % | exp {m.get('exp',0):.3f} R | PF {m.get('pf',0):.2f} | DD {m.get('dd',0):.1f} R")
    print("VEREDICTO:", "PASA a papel en vivo" if pasa else "NO PASA")
    print("Informe: informe_backtest.html · Operaciones: operaciones_backtest.csv")


if __name__ == "__main__":
    main()
