"""
回测引擎 - 数据结构定义

Trade / BacktestResult 供 agent_backtest.py 使用。
实际回测逻辑在 agent_backtest.py 中（DecisionParams 驱动）。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: int              # 1=long, -1=short
    margin: float               # 保证金
    leverage: int
    quantity: float = 0.0
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    gross_pnl: float = 0.0
    entry_fee_paid: float = 0.0
    exit_fee_paid: float = 0.0
    funding_fee_paid: float = 0.0
    exit_reason: str = ""
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trailing_high: Optional[float] = None
    trailing_low: Optional[float] = None
    entry_time: Optional[str] = None   # ISO 8601 timestamp
    exit_time: Optional[str] = None    # ISO 8601 timestamp
    holding_bars: int = 0
    holding_hours: float = 0.0


@dataclass
class BacktestResult:
    backend: str = "local_backtest"
    detail_level: str = "trade_level"
    execution_profile: str = ""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    initial_equity: float = 0.0
    ending_equity: float = 0.0
    net_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    regime_performance: dict = field(default_factory=dict)
    blowup_count: int = 0
    liquidation_count: int = 0
    total_deposited: float = 0.0
    gross_profit_pnl: float = 0.0
    gross_loss_pnl: float = 0.0
    trading_fee_paid: float = 0.0
    funding_fee_paid: float = 0.0
    fill_count: int = 0
    long_trade_count: int = 0
    short_trade_count: int = 0
    avg_holding_bars: float = 0.0
    avg_holding_hours: float = 0.0
    open_positions: list[Trade] = field(default_factory=list)
    # Individual fills: list of dicts with {side, qty, price, is_liquidation}.
    # Populated by run_backtest for evolve-merge aggregation parity with backend.
    fills: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "detail_level": self.detail_level,
            "execution_profile": self.execution_profile,
            "initial_equity": round(self.initial_equity, 4),
            "ending_equity": round(self.ending_equity, 4),
            "net_pnl": round(self.net_pnl, 4),
            "total_return": round(self.total_return, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "blowup_count": self.blowup_count,
            "liquidation_count": self.liquidation_count,
            "total_deposited": round(self.total_deposited, 2),
            "gross_profit_pnl": round(self.gross_profit_pnl, 4),
            "gross_loss_pnl": round(self.gross_loss_pnl, 4),
            "trading_fee_paid": round(self.trading_fee_paid, 4),
            "funding_fee_paid": round(self.funding_fee_paid, 4),
            "fill_count": self.fill_count,
            "long_trade_count": self.long_trade_count,
            "short_trade_count": self.short_trade_count,
            "avg_holding_bars": round(self.avg_holding_bars, 4),
            "avg_holding_hours": round(self.avg_holding_hours, 4),
            "regime_performance": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.regime_performance.items()
            },
        }
