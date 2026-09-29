"""
DERIV SENTINEL - INDEPENDENT CAPITAL-PRESERVATION BOT (DEMO ONLY)
===================================================================

Not derived from Harness 4/6/7. Built from a single premise, taken directly
from your Research Analyzer's result on 713,820 R_100 ticks: digit outcomes
on Deriv's volatility indices show no usable pattern (uniform distribution,
no transition dependency at lags 1-3, no streak effect, no price-direction
link, nothing survived correction or out-of-sample re-testing).

Given that premise, there is exactly one honest design for a bot whose job is
to survive rather than to "beat" the game:

  1. Never guess the next digit. There is nothing to guess.
  2. The only real number in this problem is each contract's RTP (return to
     player), which is fixed by Deriv's pricing, not by history. Always take
     whichever available contract currently has the highest real RTP.
  3. Trade rarely. Expected loss = turnover x (1 - RTP). Cutting how often you
     trade is the only lever that actually reduces expected loss, since there
     is no smarter way to place any single trade.
  4. Flat stakes, always. No martingale, no "recovery" stake increase -
     compounding a stake after a loss cannot fix a negative-expectation game;
     it only makes the eventual bad run larger.
  5. Hard, absolute stops: a session loss cap, a losing-streak cap, and a
     trade-count cap. Once any is hit, the bot stops placing orders - it can
     keep watching the market for logging purposes, but it will not spend
     another cent.
  6. Be honest about the odds out loud: before you ever start, and after every
     trade, this bot prints an estimate of how many trades you can expect to
     survive at the current stake and RTP before the session cap is reached
     (in expectation - real sessions vary a lot around that number).

This bot will not make money for you on average. Nothing legitimate can, on
these instruments, per your own research. What it can do is spend your
demo balance as slowly and as visibly as possible, which is the actual goal
you described.

STAKING: Oscar's Grind
-----------------------
Oscar's Grind is a bet-SIZING system, not a prediction system - it changes how
big each stake is, never which contract wins. It cannot turn a negative-EV
contract positive; nothing can. What it does:

  - Start each "cycle" at the base unit (= `stake`), with cycle profit = 0.
  - After a LOSS: stake stays exactly the same for the next bet. (This is
    the opposite of martingale, which doubles after a loss - this bot still
    never does that.)
  - After a WIN that has not yet reached the cycle's target profit (one base
    unit, by default): stake goes UP by one base unit for the next bet.
  - After a WIN that reaches the target: the cycle resets to the base stake
    and zero cycle profit, and a new cycle begins.

Classic Oscar's Grind is designed for close-to-even-money bets, where a win
and a loss are roughly the same size. OVER 1 is not that: it wins about 80%
of the time, but a win only pays about 23% of the stake while a loss costs
the full stake. That mismatch matters here specifically because Oscar's
Grind ratchets the stake UP after each win that doesn't hit the target - and
with OVER 1, most wins don't hit the target on their own (a $1 win only
nets ~$0.23 against a $1 target), so the system tends to raise the stake
several times per cycle purely because wins are common, right up until the
~20%-probability loss lands at that now-larger size. In other words: the
specific danger with this contract is not "one huge loss undoes many small
wins" (true of OVER 1 regardless of staking), it's that Oscar's Grind will
often have QUIETLY INCREASED the size of that eventual loss by the time it
happens, because the grind's own logic keeps raising stakes on an
easy-to-win, hard-to-profit-from contract. `oscars_max_stake` puts a hard
ceiling on this, and `oscars_max_cycle_trades` stops the bot if a single
cycle runs unreasonably long without completing. The pre-existing
`max_session_loss`, `max_losing_streak`, and `max_trades` caps still apply
underneath all of this, independent of whatever the grind is doing.

Set `staking_mode` to `"FLAT"` to disable the grind entirely and go back to
a constant stake picked by highest RTP across all contracts (the original
behaviour of this bot).

Setup
-----
Dependency (once per virtualenv):
    pip install websockets

Three layers of settings, later ones override earlier ones:
    1. The defaults in the Settings dataclass below.
    2. config.json in the current folder (or --config path), same field
       names as Settings, e.g.:
           {"symbol": "R_50", "stake": 1.0, "max_session_loss": 15.0}
       Unknown keys are logged and ignored rather than failing the run.
    3. Environment variables prefixed SENTINEL_ (e.g. SENTINEL_SYMBOL,
       SENTINEL_STAKE, SENTINEL_MAX_SESSION_LOSS) - handy for a project
       that already keeps secrets/settings in a .env / run configuration.
    4. Command-line flags (--symbol, --stake, ...) always win last.

Credentials are separate from the above: get_websocket_url() comes from your
`authenticator` module (same as your other harnesses), which is expected to
read whatever token/app-id environment variables your project already uses.
If `authenticator` isn't importable, DERIV_WS_URL is used as a fallback.
Never hard-code a token anywhere in source or in config.json.

Usage
-----
    python sentinel.py                          # demo, R_50, $1 stake
    python sentinel.py --config myconf.json --symbol R_100 --watch-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("Sentinel")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# The candidate contracts. Only "plain" digit contracts with normal payout
# behaviour: Differs (all 10 barriers), Even, Odd, and moderate Over/Under
# barriers (1-8). The most extreme Over/Under barriers (0 and 9) are excluded
# on purpose - they carry the largest payout multiples and, in practice, the
# thinnest RTP, i.e. the worst deal, precisely because they look the most
# exciting.
# =============================================================================

Spec = tuple[str, Optional[int]]


def candidate_specs() -> list[Spec]:
    specs: list[Spec] = [("DIGITDIFF", d) for d in range(10)]
    specs += [("DIGITEVEN", None), ("DIGITODD", None)]
    specs += [("DIGITOVER", b) for b in range(1, 9)]
    specs += [("DIGITUNDER", b) for b in range(1, 9)]
    return specs


def true_win_probability(ctype: str, barrier: Optional[int]) -> float:
    """
    The correct win probability under a fair, independent digit - which your
    research confirms is what Deriv's synthetic indices actually produce.
    This is a fact about arithmetic, not a prediction: e.g. DIGITOVER 4 wins
    whenever the digit is 5-9, which is 5 of 10 equally likely digits.
    """
    if ctype == "DIGITDIFF":
        return 0.9
    if ctype in ("DIGITEVEN", "DIGITODD"):
        return 0.5
    if ctype == "DIGITOVER":
        return (9 - barrier) / 10.0
    if ctype == "DIGITUNDER":
        return barrier / 10.0
    return 0.0


def label(spec: Spec) -> str:
    ctype, barrier = spec
    name = ctype.replace("DIGIT", "")
    return name if barrier is None else f"{name} {barrier}"


def extract_digit(quote: Any, decimals: int) -> int:
    return int(f"{float(quote):.{decimals}f}"[-1])


def ensure_dependencies() -> None:
    """
    Fail fast with an actionable message instead of a bare ModuleNotFoundError
    traceback partway through a session. Run once at startup, before connect().
    """
    missing = [m for m in ("websockets",) if importlib.util.find_spec(m) is None]
    if missing:
        log.error("Missing required package(s): %s", ", ".join(missing))
        log.error("Fix: run this in your project's virtualenv, then re-run the bot:")
        log.error("    %s -m pip install %s", sys.executable, " ".join(missing))
        raise FatalError(f"missing dependency: {', '.join(missing)}")


# =============================================================================
# SETTINGS
# =============================================================================

@dataclass
class Settings:
    symbol: str = "R_50"
    stake: float = 1.00
    currency: str = "USD"
    duration: int = 1
    duration_unit: str = "t"

    min_rtp: float = 0.95              # never take a worse deal than this
    repricing_seconds: float = 600.0   # how often to re-check every contract's RTP

    seconds_between_trades: float = 60.0
    loss_streak_pause_at: int = 3
    loss_streak_pause_seconds: float = 300.0

    max_session_loss: float = 15.00
    max_losing_streak: int = 6
    max_trades: int = 150
    session_win_target: Optional[float] = None

    # ---- staking ----------------------------------------------------------
    staking_mode: str = "OSCARS_GRIND"     # "FLAT" | "OSCARS_GRIND"

    # Oscar's Grind, fixed to ONE contract (its cycle math assumes a constant
    # win probability and payout; switching contracts mid-cycle would make the
    # cycle's target meaningless). Defaults to OVER 1, since that's the
    # contract in question; change via --oscars-contract or config.json.
    oscars_contract_type: str = "DIGITOVER"
    oscars_contract_barrier: Optional[int] = 1
    oscars_target_profit: Optional[float] = None   # None -> defaults to `stake` (1 unit)
    oscars_max_stake: float = 8.00                 # hard ceiling regardless of grind state
    oscars_max_cycle_trades: int = 40              # safety stop for a cycle that won't close

    watch_only: bool = False           # log/report but never place an order

    require_demo_login: bool = True
    demo_login_prefixes: tuple[str, ...] = ("VRT", "DEM", "DOT")

    request_timeout: float = 10.0
    buy_timeout: float = 6.0
    ping_interval: int = 20
    max_ask_slippage: float = 0.05
    reconcile_delay: float = 6.0
    abandon_after: float = 90.0
    report_every: float = 300.0

    data_dir: Path = Path("sentinel_data")


class FatalError(RuntimeError):
    pass


def validate_settings(cfg: Settings) -> None:
    if cfg.staking_mode not in ("FLAT", "OSCARS_GRIND"):
        raise FatalError(f"unknown staking_mode={cfg.staking_mode!r} (expected FLAT or OSCARS_GRIND)")
    if cfg.staking_mode == "OSCARS_GRIND":
        target = cfg.oscars_target_profit if cfg.oscars_target_profit is not None else cfg.stake
        if target <= 0:
            raise FatalError("oscars_target_profit must be > 0")
        if cfg.oscars_max_stake < cfg.stake:
            raise FatalError("oscars_max_stake cannot be smaller than the base stake")
        if cfg.oscars_max_cycle_trades < 1:
            raise FatalError("oscars_max_cycle_trades must be >= 1")
        ctype = cfg.oscars_contract_type
        barrier = cfg.oscars_contract_barrier
        if ctype in ("DIGITEVEN", "DIGITODD") and barrier is not None:
            raise FatalError(f"{ctype} takes no barrier, got {barrier}")
        if ctype not in ("DIGITEVEN", "DIGITODD") and barrier is None:
            raise FatalError(f"{ctype} requires oscars_contract_barrier")


def parse_contract_label(text: str) -> Spec:
    """Parse a short label like 'OVER1', 'OVER 1', 'EVEN', 'DIFFERS3' into a Spec."""
    import re
    t = text.strip().upper().replace("DIGIT", "")
    m = re.match(r"^([A-Z]+)\s*(\d+)?$", t)
    if not m:
        raise FatalError(f"could not parse contract label {text!r}")
    name, num = m.group(1), m.group(2)
    mapping = {
        "OVER": "DIGITOVER", "UNDER": "DIGITUNDER",
        "DIFF": "DIGITDIFF", "DIFFERS": "DIGITDIFF",
        "EVEN": "DIGITEVEN", "ODD": "DIGITODD",
        "MATCH": "DIGITMATCH", "MATCHES": "DIGITMATCH",
    }
    if name not in mapping:
        raise FatalError(f"unknown contract type in {text!r}")
    ctype = mapping[name]
    barrier = int(num) if num is not None else None
    if ctype in ("DIGITEVEN", "DIGITODD") and barrier is not None:
        raise FatalError(f"{name} takes no barrier ({text!r})")
    if ctype not in ("DIGITEVEN", "DIGITODD") and barrier is None:
        raise FatalError(f"{name} requires a barrier, e.g. 'OVER1' ({text!r})")
    return (ctype, barrier)


# =============================================================================
# OSCAR'S GRIND: stake sizing only. Does not touch which contract is bought,
# does not touch win probability, does not touch RTP. It only decides how many
# dollars ride on the NEXT bet of the one fixed contract it is configured for.
# =============================================================================

class OscarsGrind:
    def __init__(self, cfg: Settings) -> None:
        self.unit = cfg.stake
        self.max_stake = cfg.oscars_max_stake
        self.target = cfg.oscars_target_profit if cfg.oscars_target_profit is not None else cfg.stake
        self.max_cycle_trades = cfg.oscars_max_cycle_trades
        self.spec: Spec = (cfg.oscars_contract_type, cfg.oscars_contract_barrier)

        self.cycle_profit = 0.0
        self.current_stake = self.unit
        self.cycle_trades = 0
        self.cycles_completed = 0
        self.cycles_started = 1

    def next_stake(self) -> float:
        return round(min(self.current_stake, self.max_stake), 2)

    def cycle_exceeded(self) -> bool:
        return self.cycle_trades >= self.max_cycle_trades

    def record(self, profit: float) -> None:
        self.cycle_profit += profit
        self.cycle_trades += 1
        if profit > 0:
            if self.cycle_profit >= self.target - 1e-9:
                self.reset()
            else:
                # Win, but the cycle target was not reached: ratchet up by one
                # unit. NOTE: for a contract whose payout isn't close to 1:1
                # (like OVER 1), this step is what quietly grows the stake
                # over a cycle even though every individual LOSS leaves the
                # stake untouched - see the docstring warning above.
                self.current_stake = min(self.current_stake + self.unit, self.max_stake)
        # Loss: current_stake is deliberately left unchanged.

    def reset(self) -> None:
        self.cycles_completed += 1
        self.cycles_started += 1
        self.cycle_profit = 0.0
        self.current_stake = self.unit
        self.cycle_trades = 0

    def status(self) -> str:
        return (f"cycle #{self.cycles_started} trade #{self.cycle_trades} | "
                f"cycle P/L ${self.cycle_profit:+.2f} (target ${self.target:.2f}) | "
                f"next stake ${self.next_stake():.2f} (cap ${self.max_stake:.2f}) | "
                f"cycles completed so far: {self.cycles_completed}")


# =============================================================================
# RTP BOOK: the only "model" this bot has, and it's not a model - it's a
# lookup of numbers Deriv itself is quoting right now.
# =============================================================================

@dataclass
class Quote:
    spec: Spec
    multiple: float
    rtp: float


class RtpBook:
    def __init__(self) -> None:
        self.quotes: dict[Spec, Quote] = {}
        self.updated_at = 0.0

    def best(self, min_rtp: float) -> Optional[Quote]:
        ok = [q for q in self.quotes.values() if q.rtp >= min_rtp]
        return max(ok, key=lambda q: q.rtp) if ok else None

    def describe(self) -> str:
        if not self.quotes:
            return "not priced yet"
        best = max(self.quotes.values(), key=lambda q: q.rtp)
        worst = min(self.quotes.values(), key=lambda q: q.rtp)
        mean = sum(q.rtp for q in self.quotes.values()) / len(self.quotes)
        return (f"{len(self.quotes)} contracts priced | best {label(best.spec)} "
                f"({100*best.rtp:.1f}%) | worst {label(worst.spec)} "
                f"({100*worst.rtp:.1f}%) | mean {100*mean:.1f}%")


# =============================================================================
# LOG FILE (simple CSV, one row per settled trade)
# =============================================================================

class TradeLog:
    FIELDS = ["time", "symbol", "contract", "contract_id", "stake", "payout",
              "rtp", "result", "profit", "running_pnl", "expected_loss_total"]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS, extrasaction="ignore").writerow(
                {"time": now_iso(), **row}
            )


# =============================================================================
# SURVIVAL LEDGER: tracks real P/L and the "expected" loss the house edge
# alone predicts, so you can tell ordinary variance from an actual problem.
# =============================================================================

class SurvivalLedger:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.pnl = 0.0
        self.expected_loss = 0.0
        self.trades = 0
        self.losing_streak = 0
        self.stop_reason = ""
        self.pause_until = 0.0

    def should_stop(self) -> str:
        if self.stop_reason:
            return self.stop_reason
        if self.pnl <= -self.cfg.max_session_loss:
            self.stop_reason = f"session loss cap reached (${self.pnl:+.2f})"
        elif self.cfg.session_win_target is not None and self.pnl >= self.cfg.session_win_target:
            self.stop_reason = f"session win target reached (${self.pnl:+.2f})"
        elif self.losing_streak >= self.cfg.max_losing_streak:
            self.stop_reason = f"{self.losing_streak} losses in a row"
        elif self.trades >= self.cfg.max_trades:
            self.stop_reason = f"trade cap ({self.cfg.max_trades}) reached"
        return self.stop_reason

    def record(self, profit: float, stake: float, rtp: float) -> None:
        self.trades += 1
        self.pnl += profit
        self.expected_loss += stake * (1.0 - rtp)
        if profit < 0:
            self.losing_streak += 1
            if self.losing_streak >= self.cfg.loss_streak_pause_at:
                self.pause_until = time.time() + self.cfg.loss_streak_pause_seconds
        else:
            self.losing_streak = 0

    def expected_trades_remaining(self, rtp: float, stake: float) -> Optional[int]:
        per_trade = stake * (1.0 - rtp)
        if per_trade <= 0:
            return None
        cushion = self.cfg.max_session_loss - max(0.0, -self.pnl)
        return max(0, int(cushion / per_trade))


# =============================================================================
# THE BOT
# =============================================================================

@dataclass
class OpenPosition:
    spec: Spec
    contract_id: str
    stake: float
    payout: float
    rtp: float
    opened_at: float


class Sentinel:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.rtp_book = RtpBook()
        self.ledger = SurvivalLedger(cfg)
        self.log = TradeLog(cfg.data_dir / cfg.symbol / "trades.csv")

        self.ws = None
        self.running = False
        self._req = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._decimals = 2

        self.balance: Optional[float] = None
        self.currency = cfg.currency
        self.loginid = ""
        self.last_trade_at = 0.0
        self.busy = False
        self.reconciling = False
        self.open_position: Optional[OpenPosition] = None
        self.seen_ids: set[str] = set()

        self.grind: Optional[OscarsGrind] = (
            OscarsGrind(cfg) if cfg.staking_mode == "OSCARS_GRIND" else None
        )

    # ---- plumbing ---------------------------------------------------------------

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        t.add_done_callback(
            lambda task: (not task.cancelled() and task.exception() and
                         log.error("[TASK ERROR] %r", task.exception()))
        )
        return t

    async def _send(self, payload: dict, timeout: Optional[float] = None,
                    force: bool = False) -> dict:
        if self.ws is None:
            raise ConnectionError("not connected")
        if not self.running and not force:
            raise RuntimeError("sentinel is not running")
        self._req += 1
        req_id = str(self._req)
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            await self.ws.send(json.dumps({**payload, "req_id": req_id}))
            return await asyncio.wait_for(fut, timeout or self.cfg.request_timeout)
        finally:
            self._pending.pop(req_id, None)

    def _fail_all_pending(self) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("connection closed"))
        self._pending.clear()

    async def _reader(self, ws) -> None:
        log.info("Reader started.")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rid = msg.get("req_id")
                if rid is not None:
                    fut = self._pending.get(str(rid))
                    if fut and not fut.done():
                        fut.set_result(msg)
                if "error" in msg:
                    log.warning("[DERIV ERROR] %s", msg["error"])
                    continue
                if "tick" in msg:
                    self._on_tick(msg["tick"])
                elif "balance" in msg:
                    b = msg["balance"]
                    self.balance = as_float(b.get("balance"))
                    self.currency = b.get("currency", self.currency)
                    self.loginid = str(b.get("loginid", self.loginid))
                elif "proposal_open_contract" in msg:
                    self._spawn(self._on_contract_update(msg["proposal_open_contract"]))
            self.running = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Reader stopped: %s", exc)
            self.running = False
        finally:
            self._fail_all_pending()

    def _on_tick(self, tick: dict) -> None:
        quote = tick.get("quote")
        if quote is None:
            return
        pip = tick.get("pip_size")
        if isinstance(pip, int) and 0 <= pip <= 8:
            self._decimals = pip
        self._maybe_trade()

    # ---- pricing ----------------------------------------------------------------

    def _proposal_payload(self, spec: Spec, stake: float) -> dict:
        ctype, barrier = spec
        payload = {"proposal": 1, "amount": round(stake, 2), "basis": "stake",
                  "contract_type": ctype, "currency": self.currency,
                  "duration": self.cfg.duration, "duration_unit": self.cfg.duration_unit,
                  "underlying_symbol": self.cfg.symbol}
        if barrier is not None:
            payload["barrier"] = str(barrier)
        return payload

    async def _price(self, spec: Spec) -> Optional[Quote]:
        try:
            resp = await self._send(self._proposal_payload(spec, 1.0))
        except Exception:
            return None
        prop = resp.get("proposal")
        if "error" in resp or not prop:
            return None
        ask, payout = as_float(prop.get("ask_price")), as_float(prop.get("payout"))
        if ask <= 0 or payout <= 0:
            return None
        multiple = payout / ask
        return Quote(spec, multiple, multiple * true_win_probability(*spec))

    async def _reprice_all(self) -> None:
        specs = candidate_specs()
        for i in range(0, len(specs), 8):
            batch = specs[i:i + 8]
            results = await asyncio.gather(*(self._price(s) for s in batch),
                                           return_exceptions=True)
            for r in results:
                if isinstance(r, Quote):
                    self.rtp_book.quotes[r.spec] = r
            await asyncio.sleep(0.3)
        self.rtp_book.updated_at = time.time()
        log.info("[PRICING] %s", self.rtp_book.describe())

    # ---- connect ------------------------------------------------------------------

    async def connect(self) -> None:
        import websockets

        loop = asyncio.get_running_loop()
        try:
            from authenticator import get_websocket_url
            log.info("Requesting authenticated demo WebSocket URL...")
            url = await loop.run_in_executor(None, get_websocket_url)
        except ImportError:
            url = os.environ.get("DERIV_WS_URL", "")
            if not url:
                raise FatalError("no `authenticator` module and DERIV_WS_URL not set")
        if not url:
            raise FatalError("no WebSocket URL returned")

        self.ws = await websockets.connect(url, ping_interval=self.cfg.ping_interval,
                                           ping_timeout=self.cfg.ping_interval, close_timeout=5)
        log.info("Connected.")
        self._reader_task = asyncio.create_task(self._reader(self.ws))

        resp = await self._send({"balance": 1, "subscribe": 1})
        if "error" in resp:
            raise FatalError(f"balance failed: {resp['error']}")
        b = resp["balance"]
        self.balance = as_float(b.get("balance"))
        self.currency = b.get("currency", self.currency)
        self.loginid = str(b.get("loginid", ""))
        log.info("[ACCOUNT] loginid=%s | equity %.2f %s", self.loginid or "?",
                 self.balance, self.currency)

        if self.cfg.require_demo_login and not self.loginid.upper().startswith(
                self.cfg.demo_login_prefixes):
            raise FatalError(f"loginid '{self.loginid}' does not look like a demo account")

        await self._reprice_all()

        r = await self._send({"ticks": self.cfg.symbol, "subscribe": 1})
        if "error" in r:
            raise FatalError(f"tick subscription failed: {r['error']}")
        log.info("[TICKS] subscribed %s", self.cfg.symbol)

    # ---- decision / execution -------------------------------------------------------

    def _can_trade(self) -> tuple[bool, str]:
        if self.cfg.watch_only:
            return False, "watch-only mode"
        if not self.running:
            return False, "not running"
        reason = self.ledger.should_stop()
        if reason:
            return False, reason
        if self.busy or self.reconciling:
            return False, "order in flight"
        now = time.time()
        if now < self.ledger.pause_until:
            return False, f"losing-streak pause ({self.ledger.pause_until - now:.0f}s left)"
        if now - self.last_trade_at < self.cfg.seconds_between_trades:
            return False, "throttle"
        return True, "ok"

    def _maybe_trade(self) -> None:
        ok, _why = self._can_trade()
        if not ok:
            return
        if self.grind is not None:
            if self.grind.cycle_exceeded():
                self.ledger.stop_reason = (
                    f"Oscar's Grind cycle #{self.grind.cycles_started} ran "
                    f"{self.grind.cycle_trades} trades without hitting its target "
                    f"(cap {self.grind.max_cycle_trades}) - stopping for safety"
                )
                return
            spec = self.grind.spec
            stake = self.grind.next_stake()
        else:
            quote = self.rtp_book.best(self.cfg.min_rtp)
            if quote is None:
                return
            spec = quote.spec
            stake = self.cfg.stake
        self.busy = True
        self.last_trade_at = time.time()
        self._spawn(self._place(spec, stake))

    async def _place(self, spec: Spec, stake: float) -> None:
        opened = handed_off = False
        try:
            resp = await self._send(self._proposal_payload(spec, stake))
            prop = resp.get("proposal")
            if "error" in resp or not prop:
                log.warning("[EXEC] proposal failed for %s: %s", label(spec),
                           resp.get("error"))
                return
            ask, payout = as_float(prop.get("ask_price")), as_float(prop.get("payout"))
            pid = str(prop.get("id", ""))
            if ask <= 0 or payout <= 0 or not pid:
                return
            live_rtp = (payout / ask) * true_win_probability(*spec)
            if live_rtp < self.cfg.min_rtp:
                log.info("[EXEC] skip %s: live RTP %.1f%% no longer qualifies",
                         label(spec), 100 * live_rtp)
                return
            if ask - stake > self.cfg.max_ask_slippage:
                log.info("[EXEC] skip %s: ask slippage", label(spec))
                return

            try:
                buy = await self._send({"buy": pid, "price": round(ask, 2)},
                                       timeout=self.cfg.buy_timeout)
            except asyncio.TimeoutError:
                log.warning("[EXEC] buy timed out; reconciling instead of guessing.")
                self.reconciling = True
                handed_off = True
                self._spawn(self._reconcile(time.time()))
                return
            if "error" in buy or not buy.get("buy"):
                log.warning("[EXEC] buy failed: %s", buy.get("error"))
                return
            cid = str(buy["buy"].get("contract_id", ""))
            if not cid:
                return

            self.seen_ids.add(cid)
            self.open_position = OpenPosition(spec, cid, ask, payout, live_rtp, time.time())
            opened = True
            runway = self.ledger.expected_trades_remaining(live_rtp, ask)
            grind_txt = f" | {self.grind.status()}" if self.grind is not None else ""
            log.info("[BUY] %s | id=%s | stake=$%.2f payout=$%.2f RTP=%.1f%% | "
                     "expected ~%s more trades before session cap%s",
                     label(spec), cid, ask, payout, 100 * live_rtp,
                     runway if runway is not None else "n/a", grind_txt)
            await self._send({"proposal_open_contract": 1, "contract_id": int(cid),
                             "subscribe": 1})
        except Exception as exc:
            log.warning("[EXEC] error: %s: %s", type(exc).__name__, exc)
        finally:
            if not opened and not handed_off:
                self.busy = False

    async def _on_contract_update(self, poc: dict) -> None:
        pos = self.open_position
        cid = str(poc.get("contract_id", ""))
        if pos is None or not cid or pos.contract_id != cid:
            return
        status = str(poc.get("status", "")).lower()
        closed = (bool(poc.get("is_sold", 0)) or bool(poc.get("is_expired", 0))
                  or status in {"sold", "expired", "closed", "won", "lost"})
        if not closed:
            return

        profit = (as_float(poc.get("profit")) if poc.get("profit") is not None
                  else as_float(poc.get("sell_price")) - pos.stake)
        self.open_position = None
        self.busy = False
        self.ledger.record(profit, pos.stake, pos.rtp)
        if self.grind is not None:
            self.grind.record(profit)
        result = "WON" if profit > 0 else "LOST"
        grind_txt = f" | {self.grind.status()}" if self.grind is not None else ""
        log.info("[SETTLED] %s | %s | profit=$%+.2f | session P/L=$%+.2f | "
                 "expected loss so far=$%.2f | losing streak=%d%s",
                 label(pos.spec), result, profit, self.ledger.pnl,
                 self.ledger.expected_loss, self.ledger.losing_streak, grind_txt)
        self.log.write({
            "symbol": self.cfg.symbol, "contract": label(pos.spec), "contract_id": cid,
            "stake": pos.stake, "payout": pos.stake + profit, "rtp": round(pos.rtp, 4),
            "result": result, "profit": profit, "running_pnl": round(self.ledger.pnl, 4),
            "expected_loss_total": round(self.ledger.expected_loss, 4),
        })
        self.ledger.should_stop()

    async def _reconcile(self, since_ts: float) -> None:
        self.reconciling = True
        try:
            await asyncio.sleep(self.cfg.reconcile_delay)
            resp = await self._send(
                {"profit_table": 1, "limit": 20, "sort": "DESC", "description": 0}, force=True)
            for tx in (resp.get("profit_table") or {}).get("transactions", []):
                cid = str(tx.get("contract_id", ""))
                if not cid or cid in self.seen_ids:
                    continue
                if as_float(tx.get("purchase_time")) < since_ts - 2:
                    continue
                buy_p, sell_p = as_float(tx.get("buy_price")), as_float(tx.get("sell_price"))
                self.seen_ids.add(cid)
                self.ledger.record(sell_p - buy_p, buy_p, 0.0)
                log.warning("[RECONCILED] untracked contract %s | P/L $%+.2f", cid, sell_p - buy_p)
        except Exception as exc:
            log.warning("[RECONCILE] failed: %s: %s", type(exc).__name__, exc)
        finally:
            self.reconciling = False
            self.busy = False

    async def _watchdog(self) -> None:
        while self.running:
            await asyncio.sleep(5)
            pos = self.open_position
            if pos and time.time() - pos.opened_at > self.cfg.abandon_after:
                log.error("[WATCHDOG] contract %s unresolved; stopping trading.",
                         pos.contract_id)
                self.open_position = None
                self.busy = False
                self.ledger.stop_reason = "unresolved contract"

    async def _periodic(self) -> None:
        last_report = last_price = time.time()
        while self.running:
            await asyncio.sleep(1)
            now = time.time()
            if now - last_price >= self.cfg.repricing_seconds:
                last_price = now
                try:
                    await self._reprice_all()
                except Exception as exc:
                    log.warning("[PRICING] refresh failed: %s", exc)
            if now - last_report >= self.cfg.report_every:
                last_report = now
                self._report()

    def _report(self) -> None:
        log.info("-" * 90)
        log.info("[STATUS] %s", self.rtp_book.describe())
        log.info("[SURVIVAL] session P/L $%+.2f | expected loss to date $%.2f | "
                 "trades %d | losing streak %d | %s", self.ledger.pnl,
                 self.ledger.expected_loss, self.ledger.trades, self.ledger.losing_streak,
                 self.ledger.stop_reason or "still trading")
        if self.grind is not None:
            q = self.rtp_book.quotes.get(self.grind.spec)
            rtp = q.rtp if q is not None else true_win_probability(*self.grind.spec)
            n = self.ledger.expected_trades_remaining(rtp, self.grind.next_stake())
            log.info("[GRIND] %s | fixed contract %s | expect ~%s more trades before "
                     "your loss cap at the CURRENT grind stake (changes as the grind moves)",
                     self.grind.status(), label(self.grind.spec),
                     n if n is not None else "n/a")
        else:
            best = self.rtp_book.best(self.cfg.min_rtp)
            if best is not None:
                n = self.ledger.expected_trades_remaining(best.rtp, self.cfg.stake)
                log.info("[FORECAST] best deal right now: %s (RTP %.1f%%) -> expect ~%s more "
                         "trades before hitting your loss cap (average case only)",
                         label(best.spec), 100 * best.rtp, n if n is not None else "n/a")
        log.info("-" * 90)

    async def _session(self) -> None:
        self.running = True
        tasks: list[asyncio.Task] = []
        try:
            await self.connect()
            tasks = [asyncio.create_task(self._periodic()), asyncio.create_task(self._watchdog())]
            log.info("=" * 90)
            log.info("SENTINEL RUNNING | %s | staking=%s | stake=$%.2f | watch_only=%s | min RTP=%.1f%%",
                     self.cfg.symbol, self.cfg.staking_mode, self.cfg.stake, self.cfg.watch_only,
                     100 * self.cfg.min_rtp)
            if self.grind is not None:
                log.info("No digit prediction of any kind. Staking = Oscar's Grind on a FIXED "
                         "contract (%s); stake ratchets up one unit per non-terminal win, holds "
                         "flat after a loss, resets after a completed cycle. Trades throttled to "
                         "one per %.0fs.", label(self.grind.spec), self.cfg.seconds_between_trades)
                q = self.rtp_book.quotes.get(self.grind.spec)
                rtp = q.rtp if q is not None else true_win_probability(*self.grind.spec)
                n0 = self.ledger.expected_trades_remaining(rtp, self.grind.next_stake())
                log.info("[FORECAST] starting point: %s at RTP %.1f%% (base stake $%.2f, cap "
                         "$%.2f) -> expect ~%s trades before the $%.2f loss cap, on average, "
                         "AT THE CURRENT STAKE (the grind's own stake changes will move this).",
                         label(self.grind.spec), 100 * rtp, self.grind.unit, self.grind.max_stake,
                         n0 if n0 is not None else "n/a", self.cfg.max_session_loss)
            else:
                log.info("No digit prediction of any kind. Contract choice = highest currently "
                         "quoted RTP. Trades throttled to one per %.0fs.",
                         self.cfg.seconds_between_trades)
                best = self.rtp_book.best(self.cfg.min_rtp)
                if best:
                    n0 = self.ledger.expected_trades_remaining(best.rtp, self.cfg.stake)
                    log.info("[FORECAST] starting point: %s at RTP %.1f%% -> expect ~%s trades "
                             "before the $%.2f loss cap, on average.", label(best.spec),
                             100 * best.rtp, n0 if n0 is not None else "n/a",
                             self.cfg.max_session_loss)
            log.info("=" * 90)
            while self.running:
                await asyncio.sleep(1)
                if self._reader_task is not None and self._reader_task.done():
                    break
        except FatalError as exc:
            log.error("[FATAL] %s", exc)
        except Exception as exc:
            log.exception("Session error: %s", exc)
        finally:
            self.running = False
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._reader_task is not None and not self._reader_task.done():
                self._reader_task.cancel()
                await asyncio.gather(self._reader_task, return_exceptions=True)
            self._fail_all_pending()
            if self.ws is not None:
                try:
                    await self.ws.close()
                except Exception:
                    pass

    async def run(self) -> None:
        try:
            await self._session()
        finally:
            self._report()
            log.info("Sentinel stopped.")


# =============================================================================
# CONFIG LOADING: defaults -> config.json -> SENTINEL_* env vars -> CLI flags
# =============================================================================

_ENV_PREFIX = "SENTINEL_"
_FIELD_TYPES = {f.name: f.type for f in fields(Settings)}


def _coerce(name: str, raw: str):
    """Best-effort cast of a config.json / env-var string to the field's type."""
    target = _FIELD_TYPES.get(name, "str")
    try:
        if target in ("float", "Optional[float]"):
            return float(raw)
        if target in ("int",):
            return int(raw)
        if target in ("bool",):
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        if target in ("Path",):
            return Path(raw)
    except (TypeError, ValueError):
        log.warning("Could not parse %s=%r as %s; ignoring.", name, raw, target)
        return None
    return raw


def load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read config file %s: %s (ignoring it)", path, exc)
        return {}
    known = set(_FIELD_TYPES)
    unknown = set(raw) - known
    if unknown:
        log.warning("Ignoring unknown key(s) in %s: %s", path, sorted(unknown))
    out = {}
    for key, value in raw.items():
        if key not in known:
            continue
        out[key] = _coerce(key, value) if isinstance(value, str) else value
    log.info("Loaded %d setting(s) from %s.", len(out), path)
    return out


def load_env_overrides() -> dict[str, Any]:
    out = {}
    for name in _FIELD_TYPES:
        env_key = _ENV_PREFIX + name.upper()
        if env_key in os.environ:
            value = _coerce(name, os.environ[env_key])
            if value is not None:
                out[name] = value
    if out:
        log.info("Applied %d override(s) from %s* environment variables: %s",
                 len(out), _ENV_PREFIX, sorted(out))
    return out


def build_settings(args: argparse.Namespace) -> Settings:
    cfg = Settings()
    cfg = replace(cfg, **load_config_file(Path(args.config)))
    cfg = replace(cfg, **load_env_overrides())

    # CLI flags win last, but only apply the ones the user actually passed
    # (argparse defaults would otherwise silently overwrite config/env values).
    cli_overrides = {}
    if args.symbol is not None:
        cli_overrides["symbol"] = args.symbol
    if args.stake is not None:
        cli_overrides["stake"] = args.stake
    if args.min_rtp is not None:
        cli_overrides["min_rtp"] = args.min_rtp
    if args.seconds_between_trades is not None:
        cli_overrides["seconds_between_trades"] = args.seconds_between_trades
    if args.max_loss is not None:
        cli_overrides["max_session_loss"] = args.max_loss
    if args.max_losing_streak is not None:
        cli_overrides["max_losing_streak"] = args.max_losing_streak
    if args.max_trades is not None:
        cli_overrides["max_trades"] = args.max_trades
    if args.watch_only:
        cli_overrides["watch_only"] = True
    if args.staking_mode is not None:
        cli_overrides["staking_mode"] = args.staking_mode
    if args.oscars_contract is not None:
        ctype, barrier = parse_contract_label(args.oscars_contract)
        cli_overrides["oscars_contract_type"] = ctype
        cli_overrides["oscars_contract_barrier"] = barrier
    if args.oscars_target_profit is not None:
        cli_overrides["oscars_target_profit"] = args.oscars_target_profit
    if args.oscars_max_stake is not None:
        cli_overrides["oscars_max_stake"] = args.oscars_max_stake
    if args.oscars_max_cycle_trades is not None:
        cli_overrides["oscars_max_cycle_trades"] = args.oscars_max_cycle_trades

    cfg = replace(cfg, **cli_overrides)
    validate_settings(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deriv Sentinel - capital-preservation bot")
    p.add_argument("--config", default="config.json",
                   help="path to a JSON settings file (default: ./config.json)")
    p.add_argument("--symbol", default=None)
    p.add_argument("--stake", type=float, default=None)
    p.add_argument("--min-rtp", type=float, default=None)
    p.add_argument("--seconds-between-trades", type=float, default=None)
    p.add_argument("--max-loss", type=float, default=None)
    p.add_argument("--max-losing-streak", type=int, default=None)
    p.add_argument("--max-trades", type=int, default=None)
    p.add_argument("--watch-only", action="store_true",
                   help="scan and log, never place an order")
    p.add_argument("--staking-mode", choices=["FLAT", "OSCARS_GRIND"], default=None)
    p.add_argument("--oscars-contract", default=None,
                   help="fixed contract for Oscar's Grind, e.g. 'OVER1', 'EVEN', 'DIFFERS3' "
                        "(default: OVER1)")
    p.add_argument("--oscars-target-profit", type=float, default=None,
                   help="cycle target profit in dollars (default: same as --stake)")
    p.add_argument("--oscars-max-stake", type=float, default=None,
                   help="hard per-trade stake ceiling for the grind (default $8.00)")
    p.add_argument("--oscars-max-cycle-trades", type=int, default=None,
                   help="stop if one cycle runs this many trades without completing (default 40)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = build_settings(args)   # already validated inside build_settings
    try:
        ensure_dependencies()
    except FatalError:
        return
    bot = Sentinel(cfg)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Stopped by user.")


if __name__ == "__main__":
    main()