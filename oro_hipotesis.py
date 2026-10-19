#!/usr/bin/env python3
"""
Ronda 2: hipótesis sacadas de la literatura (fijadas ANTES de ver resultados).

  H1  Benchmark: largo lunes apertura -> viernes cierre, todas las semanas.
  H2  Efecto viernes: largo jueves cierre -> viernes cierre.
  H3  H2 solo si la rentabilidad de 12 meses es > 0 (Bartsch et al. 2018).
  H4  Evitar lunes: largo martes apertura -> viernes cierre, con filtro 12 meses.
  H5  Regla de retroceso de la ronda 1, solo largos (referencia).

Todas: stop protector 1,5 x ATR(14), 1R = esa distancia, riesgo 1 %, costes
por activo, cierre obligatorio el último día de la semana.

Criterios (5 hipótesis -> umbral de significación corregido, Bonferroni):
  1. t-stat de la expectativa in-sample >= 2,5
  2. R por año in-sample >= 2,0   (≈ +2 % anual arriesgando 1 %)
  3. Max drawdown in-sample <= 15 R
  4. Expectativa out-of-sample > 0
  5. Positiva en GLD (mismo activo, otra fuente) y en >= 1 de plata/platino
  6. H2-H5: más R por año in-sample que el benchmark H1
"""
import argparse
import html
import math
import os

import numpy as np
import pandas as pd

import oro_backtest as ob

CRIT_T = 2.5
CRIT_RANIO = 2.0
CRIT_DD = 15.0

HIPOTESIS = {
    "H1_lun_vie": dict(dia=0, momento="open", filtro=False),
    "H2_jue_vie": dict(dia=3, momento="close", filtro=False),
    "H3_jue_vie_12m": dict(dia=3, momento="close", filtro=True),
    "H4_mar_vie_12m": dict(dia=1, momento="open", filtro=True),
    "H5_retroceso_largos": None,
}
BENCH = "H1_lun_vie"


def calendario(d, nombre, dia, momento, filtro, coste_pct, activo, atr_mult=1.5):
    o, h, l, c = (d[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    atr, r12 = d["atr"].to_numpy(), d["r12"].to_numpy()
    dow, fin = d["dow"].to_numpy(), d["fin_semana"].to_numpy()
    fechas, n = d.index, len(d)
    ops = []
    for i in range(1, n):
        if dow[i] != dia:
            continue
        ref = i if momento == "close" else i - 1      # sin mirar al futuro
        if filtro and not (r12[ref] > 0):
            continue
        if momento == "close":
            if fin[i] or i == n - 1:
                continue
            entrada, k = c[i], i + 1
        else:
            entrada, k = o[i], i
        riesgo = atr_mult * atr[ref]
        sl = entrada - riesgo
        while True:
            if k > i and o[k] <= sl:
                salida, motivo = o[k], "SL (gap)"
            elif l[k] <= sl:
                salida, motivo = sl, "SL"
            elif fin[k] or k == n - 1:
                salida, motivo = c[k], "Viernes"
            else:
                k += 1
                continue
            break
        r = (salida - entrada - entrada * coste_pct / 100) / riesgo
        dias = k - i + (1 if momento == "open" else 0)
        ops.append({
            "activo": activo, "variante": nombre, "dir": "Largo",
            "fecha_senal": fechas[ref].date(), "fecha_entrada": fechas[i].date(),
            "fecha_salida": fechas[k].date(), "entrada": round(entrada, 4),
            "sl": round(sl, 4), "tp": None, "salida": round(salida, 4),
            "motivo": motivo, "R": round(r, 4), "dias": dias,
        })
    return pd.DataFrame(ops)


def stats(ops, anios):
    if ops is None or len(ops) == 0:
        return {"n": 0}
    m = ob.metricas(ops, anios)
    rs = ops["R"].to_numpy()
    sd = rs.std(ddof=1) if len(rs) > 1 else float("nan")
    m["t"] = rs.mean() / (sd / math.sqrt(len(rs))) if sd and sd > 0 else float("nan")
    m["r_anio"] = rs.sum() / anios if anios else float("nan")
    return m


CAB = ["", "Trades", "Trades/año", "Win %", "Exp. (R)", "t-stat", "R/año", "PF", "Max DD", "CAGR (1 %)"]


def fila(nombre, m):
    if m.get("n", 0) == 0:
        return [html.escape(nombre), "0"] + ["—"] * 8
    return [html.escape(nombre), str(m["n"]), ob.f(m["por_anio"], 1), ob.f(m["win"], 1, " %"),
            ob.f(m["exp"], 3), ob.f(m["t"], 2), ob.f(m["r_anio"], 2), ob.f(m["pf"]),
            ob.f(m["dd"], 1, " R"), ob.f(m["cagr"], 1, " %")]


def evaluar(res):
    oro = res[ob.PRINCIPAL]
    bench_ranio = oro[BENCH]["is"].get("r_anio", float("nan"))
    out = {}
    for hname in HIPOTESIS:
        mi, mo = oro[hname]["is"], oro[hname]["oos"]
        ok_n = mi.get("n", 0) > 0
        gld = res["GLD (ETF)"][hname]["total"].get("exp", -1) > 0
        otros = sum(res[a][hname]["total"].get("exp", -1) > 0 for a in ("Plata (SI=F)", "Platino (PL=F)"))
        checks = [
            (f"t-stat IS ≥ {ob.f(CRIT_T,1)}", ok_n and mi["t"] >= CRIT_T, ob.f(mi.get("t"), 2)),
            (f"R/año IS ≥ {ob.f(CRIT_RANIO,1)}", ok_n and mi["r_anio"] >= CRIT_RANIO, ob.f(mi.get("r_anio"), 2)),
            (f"Max DD IS ≤ {ob.f(CRIT_DD,0)} R", ok_n and mi["dd"] <= CRIT_DD, ob.f(mi.get("dd"), 1)),
            ("Expectativa OOS > 0", mo.get("n", 0) > 0 and mo["exp"] > 0, ob.f(mo.get("exp"), 3)),
            ("Positiva en GLD y en ≥1 de plata/platino", gld and otros >= 1, f"GLD {'sí' if gld else 'no'} · {otros}/2"),
        ]
        if hname != BENCH:
            checks.append(("Bate al benchmark H1 (R/año IS)", ok_n and mi["r_anio"] > bench_ranio,
                           f"{ob.f(mi.get('r_anio'), 2)} vs {ob.f(bench_ranio, 2)}"))
        out[hname] = (all(c[1] for c in checks), checks)
    return out


def informe(res, veredictos, info, path):
    oro = res[ob.PRINCIPAL]
    h = ("<!doctype html><html lang=es><head><meta charset=utf-8>"
         "<meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Hipótesis Oro Semanal</title><style>{ob.CSS}</style></head><body><main>")
    h += "<h1>Ronda 2 · Hipótesis de la literatura · Oro</h1>"
    h += (f"<p class=mut>{html.escape(info)} · IS hasta {ob.CORTE_OOS} · OOS desde {ob.CORTE_OOS} · "
          "Solo largos · Stop 1,5×ATR · Riesgo 1 % · Cierre obligatorio el viernes</p>")

    h += "<h2>Veredicto por hipótesis</h2>"
    filas = []
    for hname, (pasa, checks) in veredictos.items():
        marcas = " ".join(f"<span class={'ok' if ok else 'ko'}>{'✔' if ok else '✘'}</span>" for _, ok, _ in checks)
        filas.append([html.escape(hname), f"<span class={'ok' if pasa else 'ko'}>{'PASA' if pasa else 'NO PASA'}</span>", marcas])
    h += ob.tabla(filas, ["Hipótesis", "Resultado", "Criterios 1-6"])

    for hname, (pasa, checks) in veredictos.items():
        h += f"<h2>{html.escape(hname)} · detalle</h2>"
        h += ob.tabla([[html.escape(t), v, f"<span class={'ok' if ok else 'ko'}>{'✔' if ok else '✘'}</span>"]
                       for t, ok, v in checks], ["Criterio", "Valor", ""])

    h += "<h2>Oro · in-sample vs out-of-sample</h2>"
    h += ob.tabla([fila(f"{k} · IS", oro[k]["is"]) for k in oro] +
                  [fila(f"{k} · OOS", oro[k]["oos"]) for k in oro], CAB)

    h += "<h2>Oro · curva de R acumulado</h2>"
    h += ob.curva_svg({k: oro[k]["ops"] for k in oro})

    h += "<h2>Todos los activos (periodo completo)</h2>"
    h += ob.tabla([fila(f"{a} · {k}", res[a][k]["total"]) for a in res for k in res[a]], CAB)

    filas = []
    for k in oro:
        ops = oro[k]["ops"]
        if len(ops):
            por = ops.groupby(pd.to_datetime(ops["fecha_entrada"]).dt.year)["R"].sum()
            filas.append([html.escape(k)] + [ob.f(por.get(y, 0), 1) for y in range(2000, 2027)])
    h += "<h2>Oro · R por año</h2>" + ob.tabla(filas, [""] + [str(y)[2:] for y in range(2000, 2027)])

    h += ("<p class=mut>Supuestos: datos diarios de Yahoo; futuros continuos sin ajustar por roll; "
          "gaps a través del stop salen a la apertura; costes por activo incluidos. "
          "Resultados pasados no garantizan resultados futuros.</p></main></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir")
    ap.add_argument("--out", default="resultados/hipotesis")
    a = ap.parse_args()
    corte = pd.Timestamp(ob.CORTE_OOS)
    res, todas, info = {}, [], []

    for activo, (ticker, coste) in ob.ACTIVOS.items():
        df = (ob.cargar_csv(os.path.join(a.csv_dir, f"{ticker.replace('=', '_')}.csv"))
              if a.csv_dir else ob.cargar_yahoo(ticker))
        r12 = df["Close"] / df["Close"].shift(252) - 1
        d = ob.indicadores(df)
        d["r12"] = r12.reindex(d.index)
        info.append(f"{activo}: {d.index[0].date()}→{d.index[-1].date()}")
        an_tot = ob.anios_entre(d)
        an_is = ob.anios_entre(d[d.index < corte])
        an_oos = ob.anios_entre(d[d.index >= corte])
        res[activo] = {}
        for hname, cfg in HIPOTESIS.items():
            if cfg is None:
                ops = ob.backtest(d, ob.Reglas(hname, solo_largos=True), coste, activo)
            else:
                ops = calendario(d, hname, cfg["dia"], cfg["momento"], cfg["filtro"], coste, activo)
            todas.append(ops)
            fe = pd.to_datetime(ops["fecha_entrada"]) if len(ops) else pd.Series([], dtype="datetime64[ns]")
            res[activo][hname] = {
                "ops": ops,
                "total": stats(ops, an_tot),
                "is": stats(ops[fe < corte] if len(ops) else ops, an_is),
                "oos": stats(ops[fe >= corte] if len(ops) else ops, an_oos),
            }
        print(f"✓ {activo}: {len(d)} días")

    os.makedirs(a.out, exist_ok=True)
    pd.concat([t for t in todas if len(t)]).to_csv(os.path.join(a.out, "operaciones_hipotesis.csv"), index=False)
    ver = evaluar(res)
    informe(res, ver, " · ".join(info), os.path.join(a.out, "informe_hipotesis.html"))

    print()
    for hname in HIPOTESIS:
        mi, mo = res[ob.PRINCIPAL][hname]["is"], res[ob.PRINCIPAL][hname]["oos"]
        print(f"{hname:20s} IS n={mi.get('n',0):4d} exp={mi.get('exp',0):+.3f} t={mi.get('t',0):5.2f} "
              f"R/año={mi.get('r_anio',0):+.2f} DD={mi.get('dd',0):5.1f} | OOS n={mo.get('n',0):3d} "
              f"exp={mo.get('exp',0):+.3f} | {'PASA' if ver[hname][0] else 'NO PASA'}")
        for t, ok, v in ver[hname][1]:
            print(f"    {'✔' if ok else '✘'} {t}: {v}")


if __name__ == "__main__":
    main()
