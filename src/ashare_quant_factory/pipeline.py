from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .calendar import get_trade_day_now
from .config import Settings
from .data.fetcher import DataFetcher
from .db.repository import (
    DB,
    get_kv,
    load_bars,
    save_best_strategy,
    save_recommendations,
    save_report_run,
    set_kv,
)
from .db.session import create_engine_from_settings, init_db
from .logging import get_logger
from .report.emailer import send_html_email
from .report.renderer import ReportRenderer
from .strategy.backtester import CostModel, backtest
from .strategy.genetic import GAConfig, GeneticEngine, parameter_sensitivity
from .strategy.recommender import make_recommendations
from .timeutils import combine_local, parse_hhmm, parse_yyyy_mm_dd, tz_now
from .utils.lock import file_lock

log = get_logger(__name__)


@dataclass(frozen=True)
class CycleResult:
    trade_date: str
    next_trade_date: str
    inserted_rows: int
    best_fitness: float
    report_path: str


class NightlyPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

        engine = create_engine_from_settings(settings)
        init_db(engine)
        self.db = DB(engine)

        self.fetcher = DataFetcher(settings, self.db)
        self.costs = CostModel(
            commission_bps=settings.backtest.commission_bps,
            stamp_duty_bps=settings.backtest.stamp_duty_bps,
            slippage_bps=settings.backtest.slippage_bps,
        )
        self.renderer = ReportRenderer(template_dir=Path(__file__).parent / "report" / "templates")

    def run_forever(self) -> None:
        tz = ZoneInfo(self.settings.project.timezone)
        poll_time = parse_hhmm(self.settings.schedule.poll_start_time)

        scheduler = BackgroundScheduler(timezone=tz)
        scheduler.add_job(
            self.run_cycle,
            trigger=CronTrigger(hour=poll_time.hour, minute=poll_time.minute),
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        scheduler.start()

        log.info(f"[AQF] Scheduler started (timezone={self.settings.project.timezone}).")
        self._kickstart_if_due(poll_time)

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            log.warning("[AQF] Interrupted. Exiting...")
        finally:
            scheduler.shutdown(wait=False)

    def _kickstart_if_due(self, poll_time: dt.time) -> None:
        now = tz_now(self.settings.project.timezone)
        if now.time() >= poll_time:
            # If service restarted after 20:00, run once immediately if needed
            try:
                self.run_cycle(force=True, skip_poll=False)
            except Exception as e:  # noqa: BLE001
                log.exception(f"[AQF] kickstart run_cycle failed: {e}")

    def run_cycle(self, *, force: bool = False, skip_poll: bool = False) -> CycleResult | None:
        """Nightly cycle: poll update -> incremental fetch -> GA evolve -> recommendations -> report -> email."""
        with file_lock(self.settings.lock_file):
            trade_day = get_trade_day_now(self.settings.project.timezone)
            trade_date, next_trade_date = trade_day.last, trade_day.next

            last_done = get_kv(self.db, "last_processed_trade_date")
            if last_done == trade_date:
                log.info(f"[AQF] Already processed trade_date={trade_date}. Skip.")
                return None

            poll_start = parse_hhmm(self.settings.schedule.poll_start_time)
            now = tz_now(self.settings.project.timezone)
            if (not force) and now.time() < poll_start:
                log.info(f"[AQF] Now<{poll_start}. Will wait for scheduler trigger.")
                return None

            stop_at = self._compute_stop_time(next_trade_date)

            # Poll for data availability
            log.info(f"[AQF] Polling Baostock daily update for trade_date={trade_date} ...")
            poll_interval = int(self.settings.schedule.poll_interval_seconds)
            deadline = min(stop_at - dt.timedelta(hours=1), now + dt.timedelta(hours=8))
            if skip_poll:
                log.info("[AQF] skip_poll=True -> assume data already available.")
            else:
                while True:
                    if self.fetcher.probe_daily_updated(trade_date):
                        log.info(f"[AQF] Detected daily update available for {trade_date}.")
                        break
                    if tz_now(self.settings.project.timezone) >= deadline:
                        log.error(f"[AQF] Daily data not available before deadline={deadline}. Abort cycle.")
                        return None
                    log.info("[AQF] Not updated yet. Sleep %ss ...", poll_interval)
                    time.sleep(poll_interval)

            # Incremental sync
            stats = self.fetcher.sync_watchlist_incremental(trade_date)
            log.info(f"[AQF] Sync done. fetched_rows={stats.fetched_rows} inserted_rows={stats.inserted_rows}")
            set_kv(self.db, "last_sync_inserted_rows", str(stats.inserted_rows))

            # Load dataset for GA
            codes = list(self.settings.watchlist)
            dataset_full = load_bars(self.db, codes, start_date=self.settings.data.history_start)
            dataset_eval = self._maybe_sample_dataset(dataset_full)

            checkpoint = Path(self.settings.reports_dir).parent / "checkpoints" / f"{trade_date.replace('-', '')}.json"
            ga_cfg = GAConfig(
                population_size=self.settings.ga.population_size,
                elite_size=self.settings.ga.elite_size,
                crossover_rate=self.settings.ga.crossover_rate,
                mutation_rate=self.settings.ga.mutation_rate,
                workers=self.settings.ga.workers,
                seed=self.settings.ga.seed,
                checkpoint_path=str(checkpoint),
                cv_method=self.settings.ga.cv_method,
                cv_splits=self.settings.ga.cv_splits,
                cv_purge_days=self.settings.ga.cv_purge_days,
            )

            log.info(f"[AQF] GA evolve until {stop_at} ... (workers={ga_cfg.workers})")
            engine = GeneticEngine(dataset=dataset_eval, costs=self.costs, cfg=ga_cfg)
            best_g, best_s = engine.evolve_until(stop_at)

            log.info(f"[AQF] GA best fitness={best_s.fitness:.4f} genome={best_g.to_dict()}")
            sensitivity = parameter_sensitivity(
                dataset_eval,
                best_g,
                self.costs,
                cv_method=ga_cfg.cv_method,
                cv_splits=ga_cfg.cv_splits,
                cv_purge_days=ga_cfg.cv_purge_days,
            )
            best_s.metrics["parameter_sensitivity"] = sensitivity
            save_best_strategy(
                self.db,
                trade_date=trade_date,
                genome=best_g.to_dict(),
                fitness=float(best_s.fitness),
                metrics=best_s.metrics,
            )

            # Recommendations
            recos = make_recommendations(
                dataset=dataset_full,
                g=best_g,
                costs=self.costs,
                max_weight_per_symbol=self.settings.risk.max_weight_per_symbol,
            )
            save_recommendations(self.db, trade_date, recos)

            # Signals for top charts
            signals_map: dict[str, pd.DataFrame] = {}
            top_codes = (
                recos.sort_values("weight", ascending=False)
                .head(self.settings.risk.top_charts)["code"]
                .astype(str)
                .tolist()
            )
            for code in top_codes:
                df = dataset_full.get(code)
                if df is None or df.empty:
                    continue
                r = backtest(df, best_g, self.costs)
                signals_map[code] = r.signals

            # Render report
            rendered = self.renderer.render(
                trade_date=trade_date,
                next_trade_date=next_trade_date,
                strategy=best_g.to_dict(),
                ga_metrics=best_s.metrics,
                recommendations=recos,
                dataset=dataset_full,
                signals_map=signals_map,
                stability_score=int(best_s.metrics.get("stability_score", 0)),
                parameter_sensitivity=best_s.metrics.get("parameter_sensitivity", []),
                out_dir=self.settings.reports_dir,
                top_charts=self.settings.risk.top_charts,
                tz_name=self.settings.project.timezone,
            )

            # Email targets
            to_emails = list(self.settings.email.to)
            if not to_emails:
                raise RuntimeError("No email.to configured. Set email.to in config.yaml or AQF_EMAIL_TO in .env")

            subject = f"{self.settings.email.subject_prefix}{trade_date} Nightly Alpha (Next: {next_trade_date})"
            email_sent = True
            try:
                send_html_email(
                    settings=self.settings,
                    subject=subject,
                    html=rendered.html,
                    to_emails=to_emails,
                    images=rendered.images,
                )
            except Exception as e:  # noqa: BLE001
                email_sent = False
                log.exception(f"[AQF] Email send failed: {e}. Report kept at {rendered.html_path}")

            save_report_run(self.db, trade_date, str(rendered.html_path), ",".join(to_emails))
            set_kv(self.db, "last_processed_trade_date", trade_date)

            if email_sent:
                log.info(f"[AQF] Report sent. html={rendered.html_path}")
            else:
                log.warning(f"[AQF] Report generated but email not sent. html={rendered.html_path}")
            return CycleResult(
                trade_date=trade_date,
                next_trade_date=next_trade_date,
                inserted_rows=stats.inserted_rows,
                best_fitness=float(best_s.fitness),
                report_path=str(rendered.html_path),
            )

    def _compute_stop_time(self, next_trade_date: str) -> dt.datetime:
        tz_name = self.settings.project.timezone
        open_t = parse_hhmm(self.settings.schedule.market_open_time)
        open_dt = combine_local(parse_yyyy_mm_dd(next_trade_date), open_t, tz_name)
        return open_dt - dt.timedelta(minutes=int(self.settings.schedule.stop_minutes_before_open))

    def _maybe_sample_dataset(self, dataset_full: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        max_n = int(self.settings.ga.max_eval_symbols)
        if max_n <= 0 or max_n >= len(dataset_full):
            return dataset_full
        # deterministic sampling
        codes = sorted(dataset_full.keys())
        rng = random.Random(self.settings.ga.seed or 42)  # noqa: S311 - deterministic sampling
        sampled = rng.sample(codes, k=max_n)
        return {c: dataset_full[c] for c in sampled}
