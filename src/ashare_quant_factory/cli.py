from __future__ import annotations

import datetime as dt
import pandas as pd
import platform
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .data.baostock_client import BaostockSession
from .db.session import create_sqlite_engine, init_db
from .logging import setup_logging
from .pipeline import NightlyPipeline
from .report.emailer import load_gmail_credentials
from .timeutils import tz_now
from .utils.validation import normalize_baostock_code

app = typer.Typer(add_completion=False, help="AShare Quant Factory (AQF) CLI")
console = Console()


def _load(config: Path, env: Optional[Path]) -> Settings:
    return load_settings(config_path=config, env_path=env)


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml"),
    env: Optional[Path] = typer.Option(None, "--env", help="Path to .env (optional)"),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="DEBUG/INFO/WARNING/ERROR"),
):
    setup_logging(log_level)
    ctx.obj = {"config": config, "env": env}


@app.command()
def doctor(
    ctx: typer.Context,
):
    """环境自检：Baostock / DB / 邮箱配置。"""
    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)

    table = Table(title="AQF Doctor", show_lines=True)
    table.add_column("Item", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    table.add_row("Python", "OK", sys.version.replace("\n", " "))
    table.add_row("Platform", "OK", f"{platform.system()} {platform.release()}")

    # Timezone
    now_bj = tz_now(settings.project.timezone)
    table.add_row("Timezone", "OK", f"{settings.project.timezone}  now={now_bj:%Y-%m-%d %H:%M:%S}")

    # Watchlist
    try:
        wl = [normalize_baostock_code(x) for x in settings.watchlist]
        table.add_row("Watchlist", "OK", f"{len(wl)} symbols")
    except Exception as e:  # noqa: BLE001
        table.add_row("Watchlist", "FAIL", str(e))

    # Baostock
    try:
        with BaostockSession() as _:
            table.add_row("Baostock login", "OK", "login/logout success")
    except Exception as e:  # noqa: BLE001
        table.add_row("Baostock login", "FAIL", str(e))

    # DB
    try:
        engine = create_sqlite_engine(settings.db_path)
        init_db(engine)
        table.add_row("SQLite", "OK", str(settings.db_path))
    except Exception as e:  # noqa: BLE001
        table.add_row("SQLite", "FAIL", str(e))

    # Email
    try:
        creds = load_gmail_credentials()
        to_list = list(settings.email.to)
        table.add_row("Gmail creds", "OK", f"{creds.address} / to={len(to_list)}")
    except Exception as e:  # noqa: BLE001
        table.add_row("Gmail creds", "WARN", str(e))

    console.print(table)


@app.command("init-db")
def init_db_cmd(ctx: typer.Context):
    """初始化数据库（SQLite）。"""
    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)
    engine = create_sqlite_engine(settings.db_path)
    init_db(engine)
    console.print(f"[bold green]OK[/] db initialized: {settings.db_path}")


@app.command()
def run(ctx: typer.Context):
    """后台调度（建议配合 systemd）。"""
    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)
    NightlyPipeline(settings).run_forever()


@app.command("run-once")
def run_once(
    ctx: typer.Context,
    force: bool = typer.Option(True, "--force/--no-force", help="Ignore 20:00 constraint"),
    skip_poll: bool = typer.Option(True, "--skip-poll/--no-skip-poll", help="Do not wait for baostock update"),
):
    """立即跑一次夜间流水线（用于测试/手动触发）。"""
    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)
    res = NightlyPipeline(settings).run_cycle(force=force, skip_poll=skip_poll)
    if res is None:
        console.print("[yellow]No run performed (maybe already processed or not due).[/]")
    else:
        console.print(f"[bold green]DONE[/] trade_date={res.trade_date} report={res.report_path}")


@app.command()
def fetch(ctx: typer.Context):
    """仅增量拉取并入库（不跑 GA、不发邮件）。"""
    from .calendar import get_trade_day_now
    from .data.fetcher import DataFetcher
    from .db.repository import DB
    from .db.session import create_sqlite_engine, init_db

    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)

    engine = create_sqlite_engine(settings.db_path)
    init_db(engine)
    db = DB(engine)
    td = get_trade_day_now(settings.project.timezone)
    stats = DataFetcher(settings, db).sync_watchlist_incremental(td.last)
    console.print(
        f"[bold green]OK[/] fetched_rows={stats.fetched_rows} inserted_rows={stats.inserted_rows} up_to={td.last}"
    )


@app.command()
def evolve(
    ctx: typer.Context,
    max_minutes: int = typer.Option(0, "--max-minutes", help="For testing: stop after N minutes (0 means until next open-30m)"),
):
    """仅运行 GA（读取 DB 数据），保存最优策略到数据库。"""
    from .calendar import get_trade_day_now
    from .db.repository import DB, save_best_strategy
    from .db.session import create_sqlite_engine, init_db
    from .db.repository import load_bars
    from .strategy.genetic import GAConfig, GeneticEngine
    from .strategy.backtester import CostModel
    from .timeutils import combine_local, parse_hhmm, parse_yyyy_mm_dd, tz_now

    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)

    engine = create_sqlite_engine(settings.db_path)
    init_db(engine)
    db = DB(engine)

    td = get_trade_day_now(settings.project.timezone)
    trade_date, next_trade_date = td.last, td.next

    dataset = load_bars(db, settings.watchlist, start_date=settings.data.history_start)
    if not any((df is not None and not df.empty) for df in dataset.values()):
        console.print("[red]No data in DB.[/] Please run: aqf fetch")
        raise typer.Exit(code=1)

    costs = CostModel(
        commission_bps=settings.backtest.commission_bps,
        stamp_duty_bps=settings.backtest.stamp_duty_bps,
        slippage_bps=settings.backtest.slippage_bps,
    )

    tz_name = settings.project.timezone
    if max_minutes > 0:
        stop_at = tz_now(tz_name) + dt.timedelta(minutes=max_minutes)
    else:
        open_t = parse_hhmm(settings.schedule.market_open_time)
        open_dt = combine_local(parse_yyyy_mm_dd(next_trade_date), open_t, tz_name)
        stop_at = open_dt - dt.timedelta(minutes=int(settings.schedule.stop_minutes_before_open))

    checkpoint = Path(settings.reports_dir).parent / "checkpoints" / f"{trade_date.replace('-', '')}.json"
    ga_cfg = GAConfig(
        population_size=settings.ga.population_size,
        elite_size=settings.ga.elite_size,
        crossover_rate=settings.ga.crossover_rate,
        mutation_rate=settings.ga.mutation_rate,
        workers=settings.ga.workers,
        seed=settings.ga.seed,
        checkpoint_path=str(checkpoint),
    )

    console.print(f"[cyan]Evolving until[/] {stop_at} (workers={ga_cfg.workers}) ...")
    best_g, best_s = GeneticEngine(dataset, costs, ga_cfg).evolve_until(stop_at)

    save_best_strategy(db, trade_date, best_g.to_dict(), float(best_s.fitness), best_s.metrics)
    console.print(f"[bold green]BEST[/] fitness={best_s.fitness:.4f} genome={best_g.to_dict()}")
    console.print(best_s.metrics)


@app.command()
def report(
    ctx: typer.Context,
    trade_date: Optional[str] = typer.Option(None, "--trade-date", help="YYYY-MM-DD (default: last trade day)"),
    send: bool = typer.Option(True, "--send/--no-send", help="Send email via Gmail SMTP"),
):
    """仅生成并（可选）发送报告（读取 DB + 最优策略）。"""
    from .calendar import get_trade_day_now
    from .db.repository import (
        DB,
        load_best_strategy,
        load_bars,
        load_recommendations,
        save_recommendations,
        save_report_run,
    )
    from .db.session import create_sqlite_engine, init_db
    from .report.renderer import ReportRenderer
    from .strategy.backtester import CostModel, Genome, backtest
    from .strategy.recommender import make_recommendations
    from .report.emailer import send_html_email

    config: Path = ctx.obj["config"]
    env: Optional[Path] = ctx.obj["env"]
    settings = _load(config, env)

    engine = create_sqlite_engine(settings.db_path)
    init_db(engine)
    db = DB(engine)

    if trade_date is None:
        td = get_trade_day_now(settings.project.timezone)
        trade_date = td.last
        next_trade_date = td.next
    else:
        # if user specifies trade_date, still compute next by querying calendar around that date
        # easiest: use today's trade_day as placeholder for next date in subject
        td = get_trade_day_now(settings.project.timezone)
        next_trade_date = td.next

    best = load_best_strategy(db, trade_date)
    if not best:
        console.print(f"[red]No best strategy found for {trade_date}.[/] Run: aqf evolve or aqf run-once")
        raise typer.Exit(code=1)

    genome = Genome.from_dict(best["genome"])
    costs = CostModel(
        commission_bps=settings.backtest.commission_bps,
        stamp_duty_bps=settings.backtest.stamp_duty_bps,
        slippage_bps=settings.backtest.slippage_bps,
    )

    dataset = load_bars(db, settings.watchlist, start_date=settings.data.history_start)

    recos = load_recommendations(db, trade_date)
    if recos.empty:
        recos = make_recommendations(dataset, genome, costs, settings.risk.max_weight_per_symbol)
        save_recommendations(db, trade_date, recos)

    signals_map: dict[str, pd.DataFrame] = {}
    top_codes = (
        recos.sort_values("weight", ascending=False).head(settings.risk.top_charts)["code"].astype(str).tolist()
        if not recos.empty
        else []
    )
    for code in top_codes:
        df = dataset.get(code)
        if df is None or df.empty:
            continue
        r = backtest(df, genome, costs)
        signals_map[code] = r.signals

    renderer = ReportRenderer(template_dir=Path(__file__).parent / "report" / "templates")
    rendered = renderer.render(
        trade_date=trade_date,
        next_trade_date=next_trade_date,
        strategy=genome.to_dict(),
        ga_metrics=best["metrics"],
        recommendations=recos,
        dataset=dataset,
        signals_map=signals_map,
        out_dir=settings.reports_dir,
        top_charts=settings.risk.top_charts,
        tz_name=settings.project.timezone,
    )

    console.print(f"[bold green]OK[/] html={rendered.html_path}")

    if send:
        to_emails = list(settings.email.to)
        if not to_emails:
            raise RuntimeError("No email.to configured. Set email.to in config.yaml or AQF_EMAIL_TO in .env")
        subject = f"{settings.email.subject_prefix}{trade_date} Nightly Alpha"
        send_html_email(
            settings=settings,
            subject=subject,
            html=rendered.html,
            to_emails=to_emails,
            images=rendered.images,
        )
        save_report_run(db, trade_date, str(rendered.html_path), ",".join(to_emails))
        console.print("[bold green]SENT[/] email report")