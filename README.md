# oro-paper

Swing trading semanal en el oro: backtest 2000–hoy y, si pasa los criterios, paper trading en vivo.

- `oro_backtest.py`: reglas, variantes, out-of-sample, Monte Carlo.
- `resultados/`: informe HTML, operaciones CSV y resumen (los genera GitHub Actions).

Reglas base: tendencia EMA20/EMA50 (largos y cortos), entrada lun–mié tras retroceso a EMA20,
SL 1,5×ATR14, TP 2R, cierre obligatorio el viernes, 1 trade/semana, riesgo 1 %.
