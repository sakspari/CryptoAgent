import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

class DBManager:
    def __init__(self, db_path: str = "data/picks_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS picks_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                pred_pct REAL,
                volatility REAL,
                current_price_usd REAL,
                reason TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def add_pick(self, ticker: str, direction: str, pred_pct: float, 
                 volatility: float, current_price_usd: float, reason: str):
        """Log a new pick"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO picks_history 
            (ticker, direction, pred_pct, volatility, current_price_usd, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, direction, pred_pct, volatility, current_price_usd, reason))
        conn.commit()
        conn.close()
    
    def get_recent_picks(self, limit: int = 10) -> List[Dict]:
        """Get recent picks"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, direction, timestamp
            FROM picks_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def should_skip(self, ticker: str, direction: str, limit: int = 10) -> bool:
        """Check if ticker+direction combo exists in recent picks"""
        recent = self.get_recent_picks(limit)
        for pick in recent:
            if pick['ticker'] == ticker and pick['direction'] == direction:
                return True
        return False
    
    def get_history_summary(self, limit: int = 3) -> str:
        """Get formatted summary of recent picks"""
        picks = self.get_recent_picks(limit)
        if not picks:
            return "No recent picks"
        
        symbols = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "→"}
        summary = ", ".join([f"{p['ticker']}{symbols.get(p['direction'], '?')}" for p in picks])
        return f"Last {len(picks)} picks: {summary}"
