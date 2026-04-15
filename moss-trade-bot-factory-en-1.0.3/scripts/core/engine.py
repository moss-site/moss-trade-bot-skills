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
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trailing_high: Optional[float] = None
    trailing_low: Optional[float] = None
    entry_time: Optional[str] = None   # ISO 8601 timestamp
    exit_time: Optional[str] = None    # ISO 8601 timestamp


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
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
    total_deposited: float = 0.0

    def to_dict(self) -> dict:
        return {
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
            "total_deposited": round(self.total_deposited, 2),
            "regime_performance": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.regime_performance.items()
            },
        }
