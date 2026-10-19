#!/usr/bin/env python3
"""
H6: timing mensual del oro (Bartsch, Baur, Dichtl & Drobetz 2018).

Regla (fijada antes de ver resultados):
  El último día hábil de cada mes, si la rentabilidad de 12 meses del activo es > 0,
  se mantiene el activo durante el mes siguiente; si no, liquidez (0 %, conservador).
  Coste por cambio de posición: 0,10 % (spread ETC + comisión).

Criterios frente a comprar y mantener (B&H):
  1. Sharpe in-sample >= Sharpe B&H in-sample
  2. Max drawdown in-sample <= 2/3 del de B&H
  3. CAGR out-of-sample > 0
  4. Sharpe >= B&H en >= 2 de 3 activos de control (GLD, plata, platino), periodo completo

Sensibilidad (solo informativa, no se usa para elegir): lookback 6 y 9 meses.
"""
import argparse
import html
import math
import os

import numpy as np
import pandas as pd

import oro_backtest as ob

COSTE = 0.10 / 100
LOOKBACKS = {"H6_12m": 12, "sens_6m": 6, "sens_9m": 9}


def mensual(close):
    m = close.resample("ME").last().dropna()
    return m


def estrategia(m, meses):
    ret = m.pct_change()
    senal = (m / m.shift(meses) - 1 > 0).astype(float)
    pos = senal.shift(1)                       # decisión a cierre de mes -> mes siguiente
    cambios = pos.diff().abs().fillna(0)
    r = (pos * ret - cambios * COSTE).iloc[meses + 1:]
    return r, pos.iloc[meses + 1:], ret.iloc[meses + 1:]


def kpis(r):
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    anios = len(r) / 12
    cagr = eq.iloc[-1] ** (1 / anios) - 1
    vol = r.std(ddof=1) * math.sqrt(12)
    dd = (1 - eq / eq.cummax()).max()
    anual = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": (r.mean() * 12) / vol if vol else float("nan"),
            "dd": dd * 100, "peor_anio": anual.min() * 100, "anios_neg": int((anual < 0).sum()),
            "n_anios": len(anual), "final": eq.iloc[-1]}


CAB = ["", "CAGR", "Volatilidad", "Sharpe", "Max DD", "Peor año", "Años neg.", "× capital"]


def fila(nombre, k, extra=None):
    if not k:
        return [html.escape(nombre)] + ["—"] * 7
    return [html.escape(nombre), ob.f(k["cagr"], 1, " %"), ob.f(k["vol"], 1, " %"), ob.f(k["sharpe"], 2),
            ob.f(-k["dd"], 1, " %"), ob.f(k["peor_anio"], 1, " %"), f"{k['anios_neg']}/{k['n_anios']}",
            ob.f(k["final"], 2, "×")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir")
    ap.add_argument("--out", default="resultados/mensual")
    a = ap.parse_args()
    corte = pd.Timestamp(ob.CORTE_OOS)
    res, info, senales = {}, [], []

    for activo, (ticker, _) in ob.ACTIVOS.items():
        df = (ob.cargar_csv(os.path.join(a.csv_dir, f"{ticker.replace('=', '_')}.csv"))
              if a.csv_dir else ob.cargar_yahoo(ticker))
        m = mensual(df["Close"])
        info.append(f"{activo}: {m.index[0]:%Y-%m}→{m.index[-1]:%Y-%m}")
        res[activo] = {}
        for nombre, lb in LOOKBACKS.items():
            r, pos, bh = estrategia(m, lb)
            res[activo][nombre] = {
                "r": r, "pos": pos,
                "total": kpis(r), "is": kpis(r[r.index < corte]), "oos": kpis(r[r.index >= corte]),
                "invertido": pos.mean() * 100, "cambios": int(pos.diff().abs().sum()),
            }
            if nombre == "H6_12m":
                res[activo]["B&H"] = {"r": bh, "total": kpis(bh), "is": kpis(bh[bh.index < corte]),
                                      "oos": kpis(bh[bh.index >= corte])}
                if activo == ob.PRINCIPAL:
                    ult = m / m.shift(12) - 1
                    senales = [(f"{i:%Y-%m}", ult.loc[i]) for i in ult.index[-6:]]
        print(f"✓ {activo}: {len(m)} meses")

    oro = res[ob.PRINCIPAL]
    h6, bh = oro["H6_12m"], oro["B&H"]
    ctrl = sum(res[x]["H6_12m"]["total"].get("sharpe", -9) >= res[x]["B&H"]["total"].get("sharpe", 9)
               for x in ob.ACTIVOS if x != ob.PRINCIPAL)
    checks = [
        ("Sharpe IS ≥ B&H", h6["is"]["sharpe"] >= bh["is"]["sharpe"], f"{ob.f(h6['is']['sharpe'],2)} vs {ob.f(bh['is']['sharpe'],2)}"),
        ("Max DD IS ≤ 2/3 de B&H", h6["is"]["dd"] <= bh["is"]["dd"] * 2 / 3, f"{ob.f(-h6['is']['dd'],1)} % vs {ob.f(-bh['is']['dd'],1)} %"),
        ("CAGR OOS > 0", h6["oos"].get("cagr", -1) > 0, ob.f(h6["oos"].get("cagr"), 1, " %")),
        ("Sharpe ≥ B&H en ≥ 2 de 3 controles", ctrl >= 2, f"{ctrl}/3"),
    ]
    pasa = all(c[1] for c in checks)

    # informe
    h = ("<!doctype html><html lang=es><head><meta charset=utf-8>"
         "<meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Oro Timing Mensual</title><style>{ob.CSS}</style></head><body><main>")
    h += "<h1>H6 · Timing mensual del oro (12 meses)</h1>"
    h += f"<p class=mut>{html.escape(' · '.join(info))} · IS hasta {ob.CORTE_OOS} · coste 0,10 % por cambio · liquidez al 0 %</p>"
    h += f"<h2>Veredicto</h2><div class=ver><span class={'ok' if pasa else 'ko'}>{'PASA' if pasa else 'NO PASA'}</span></div><br>"
    h += ob.tabla([[html.escape(t), v, f"<span class={'ok' if ok else 'ko'}>{'✔' if ok else '✘'}</span>"] for t, ok, v in checks],
                  ["Criterio", "Valor", ""])
    h += "<h2>Oro · H6 frente a comprar y mantener</h2>"
    h += ob.tabla([fila(f"{n} · {p}", oro[n][p]) for n in ("H6_12m", "B&H") for p in ("total", "is", "oos")], CAB)
    h += f"<p class=mut>Tiempo invertido: {ob.f(h6['invertido'],0,' %')} · Cambios de posición: {h6['cambios']} en {len(h6['r'])} meses</p>"
    h += "<h2>Controles (periodo completo)</h2>"
    h += ob.tabla([fila(f"{x} · {n}", res[x][n]["total"]) for x in ob.ACTIVOS if x != ob.PRINCIPAL for n in ("H6_12m", "B&H")], CAB)
    h += "<h2>Sensibilidad del lookback (oro, solo informativa)</h2>"
    h += ob.tabla([fila(n, oro[n]["total"]) for n in LOOKBACKS], CAB)
    anual = pd.DataFrame({k: (1 + oro[k]["r"]).groupby(oro[k]["r"].index.year).prod() - 1 for k in ("H6_12m", "B&H")})
    h += "<h2>Oro · rentabilidad por año</h2>" + ob.tabla(
        [[str(y), ob.f(r["H6_12m"] * 100, 1, " %"), ob.f(r["B&H"] * 100, 1, " %")] for y, r in anual.iterrows()],
        ["Año", "H6", "B&H"])
    h += "<h2>Señal reciente (oro)</h2>" + ob.tabla(
        [[m_, ob.f(v * 100, 1, " %"), "Invertido" if v > 0 else "Liquidez"] for m_, v in senales], ["Mes", "Rent. 12m", "Señal"])
    h += "<p class=mut>Futuros continuos sin ajustar por roll. Resultados pasados no garantizan resultados futuros.</p></main></body></html>"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "informe_mensual.html"), "w", encoding="utf-8") as fh:
        fh.write(h)

    print()
    for n in ("H6_12m", "B&H", "sens_6m", "sens_9m"):
        for p in ("total", "is", "oos"):
            k = oro[n][p]
            print(f"{n:8s} {p:5s} CAGR={k['cagr']:6.2f}% vol={k['vol']:5.1f}% Sharpe={k['sharpe']:.2f} DD={k['dd']:5.1f}% peor={k['peor_anio']:6.1f}% neg={k['anios_neg']}/{k['n_anios']}")
    for x in ob.ACTIVOS:
        if x != ob.PRINCIPAL:
            a_, b_ = res[x]["H6_12m"]["total"], res[x]["B&H"]["total"]
            print(f"CTRL {x:15s} H6 CAGR={a_['cagr']:.2f}% Sh={a_['sharpe']:.2f} DD={a_['dd']:.1f}% | B&H CAGR={b_['cagr']:.2f}% Sh={b_['sharpe']:.2f} DD={b_['dd']:.1f}%")
    print(f"Invertido {h6['invertido']:.0f}% · cambios {h6['cambios']} · señal reciente {senales}")
    for t, ok, v in checks:
        print(f"  {'✔' if ok else '✘'} {t}: {v}")
    print("VEREDICTO:", "PASA" if pasa else "NO PASA")


if __name__ == "__main__":
    main()
