from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..timeutils import tz_now
from .charts import ChartSpec, make_candlestick_with_signals, make_risk_gauge


@dataclass(frozen=True)
class InlineImage:
    cid: str
    content: bytes
    mime_subtype: str = "png"


@dataclass(frozen=True)
class RenderedReport:
    html: str
    images: list[InlineImage]
    html_path: Path


def _safe_cid(s: str) -> str:
    # cid must be ascii-ish
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)


class ReportRenderer:
    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.template = self.env.get_template("email_report.html.j2")

    def render(
        self,
        *,
        trade_date: str,
        next_trade_date: str,
        strategy: dict[str, Any],
        ga_metrics: dict[str, Any],
        recommendations: pd.DataFrame,
        dataset: dict[str, pd.DataFrame],
        signals_map: dict[str, pd.DataFrame] | None,
        out_dir: Path,
        top_charts: int = 6,
        tz_name: str = "Asia/Shanghai",
    ) -> RenderedReport:
        out_dir.mkdir(parents=True, exist_ok=True)

        recos = recommendations.copy()
        if not recos.empty and "weight" in recos.columns:
            recos = recos.sort_values(["weight"], ascending=False)
        top = recos.head(top_charts) if not recos.empty else recos

        images: list[InlineImage] = []

        # Risk gauge score: weighted average risk_score of BUY/HOLD positions
        gauge_score = 0
        if not recos.empty and "risk_score" in recos.columns and "weight" in recos.columns:
            x = recos.dropna(subset=["risk_score"]).copy()
            if not x.empty:
                w = x["weight"].astype(float).to_numpy()
                r = x["risk_score"].astype(float).to_numpy()
                if w.sum() > 0:
                    gauge_score = int(round((w * r).sum() / w.sum()))
                else:
                    gauge_score = int(round(float(r.mean())))
        gauge_cid = _safe_cid(f"gauge_{trade_date}")
        gauge_png = make_risk_gauge(gauge_score, title="Portfolio Risk")
        images.append(InlineImage(cid=gauge_cid, content=gauge_png))

        charts: list[dict[str, Any]] = []
        for _, row in top.iterrows():
            code = str(row["code"])
            df = dataset.get(code)
            if df is None or df.empty:
                continue

            signals = pd.DataFrame()
            if signals_map and code in signals_map:
                signals = signals_map[code]

            cid = _safe_cid(f"chart_{trade_date}_{code}")
            png = make_candlestick_with_signals(
                df=df,
                signals=signals,
                title=f"{code}  K线与信号",
                spec=ChartSpec(),
            )
            if not png:
                continue
            images.append(InlineImage(cid=cid, content=png))
            charts.append(
                {
                    "code": code,
                    "cid": cid,
                    "caption": f"{row.get('action','')}  weight={float(row.get('weight',0.0)):.1%}",
                }
            )

        title = "AQF Nightly Alpha Report"
        generated_at = tz_now(tz_name).strftime("%Y-%m-%d %H:%M:%S")
        html = self.template.render(
            title=title,
            trade_date=trade_date,
            next_trade_date=next_trade_date,
            generated_at=generated_at,
            strategy=strategy,
            ga_metrics=ga_metrics,
            recommendations=recos.to_dict(orient="records") if not recos.empty else [],
            charts=charts,
            gauge_cid=gauge_cid,
        )

        ymd = trade_date.replace("-", "")
        html_path = out_dir / f"{ymd}_aqf_report.html"
        html_path.write_text(html, encoding="utf-8")

        return RenderedReport(html=html, images=images, html_path=html_path)
