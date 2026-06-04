"""
Store Intelligence — POS Transaction Correlator

Reads pos_transactions.csv. For each transaction, finds all visitor sessions
where the visitor was in the billing zone within a 5-minute window before
the transaction timestamp (correlated by store_id + time window).

Marks those sessions as converted=true.
Emits BILLING_QUEUE_ABANDON for visitors who entered the billing zone
but no transaction followed within 10 minutes.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class POSCorrelator:
    """
    Correlates POS transactions with visitor movement data.
    
    Uses a 5-minute window to match billing zone visits
    with purchase transactions.
    """

    CORRELATION_WINDOW_MIN = 5   # Minutes before transaction to look for visitors
    ABANDON_TIMEOUT_MIN = 10     # Minutes without purchase = abandonment

    def __init__(self, pos_csv_path: str = "pos_transactions.csv"):
        """
        Args:
            pos_csv_path: Path to POS transactions CSV file
        """
        self.transactions: list[dict] = []
        self._load_transactions(pos_csv_path)

    def _load_transactions(self, csv_path: str):
        """Load POS transactions from CSV."""
        path = Path(csv_path)
        if not path.exists():
            logger.warning(f"POS CSV not found: {csv_path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.transactions.append({
                        "transaction_id": row.get("transaction_id", ""),
                        "store_id": row.get("store_id", ""),
                        "timestamp": datetime.fromisoformat(
                            row.get("timestamp", "").replace("Z", "+00:00")
                        ),
                        "basket_value_inr": float(
                            row.get("basket_value_inr", 0)
                        ),
                    })

            logger.info(f"Loaded {len(self.transactions)} POS transactions")

        except Exception as e:
            logger.error(f"Failed to load POS transactions: {e}")

    def correlate_visitor(
        self,
        store_id: str,
        visitor_id: str,
        billing_zone_enter_time: datetime,
        billing_zone_exit_time: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Check if a visitor's billing zone visit correlates with a POS transaction.
        
        Args:
            store_id: Store identifier
            visitor_id: Visitor identifier
            billing_zone_enter_time: When visitor entered billing zone
            billing_zone_exit_time: When visitor left billing zone (optional)
            
        Returns:
            Matched transaction dict, or None if no match
        """
        window_start = billing_zone_enter_time - timedelta(
            minutes=self.CORRELATION_WINDOW_MIN
        )
        window_end = (
            billing_zone_exit_time or billing_zone_enter_time
        ) + timedelta(minutes=self.CORRELATION_WINDOW_MIN)

        for txn in self.transactions:
            if txn["store_id"] != store_id:
                continue
            if window_start <= txn["timestamp"] <= window_end:
                logger.info(
                    f"POS correlation found: visitor={visitor_id}, "
                    f"txn={txn['transaction_id']}, "
                    f"value={txn['basket_value_inr']}"
                )
                return txn

        return None

    def check_abandonment(
        self,
        store_id: str,
        billing_zone_enter_time: datetime,
        current_time: datetime,
    ) -> bool:
        """
        Check if a billing zone visit is an abandonment.
        
        A visitor is considered to have abandoned if they've been
        in the billing zone for > ABANDON_TIMEOUT_MIN without a
        matching POS transaction.
        """
        time_in_zone = (current_time - billing_zone_enter_time).total_seconds() / 60

        if time_in_zone < self.ABANDON_TIMEOUT_MIN:
            return False

        # Check for any transaction in the window
        window_start = billing_zone_enter_time
        window_end = current_time

        for txn in self.transactions:
            if txn["store_id"] != store_id:
                continue
            if window_start <= txn["timestamp"] <= window_end:
                return False  # Transaction found — not abandoned

        return True

    def get_transactions_for_store(
        self,
        store_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict]:
        """Get all transactions for a store within a time range."""
        results = []
        for txn in self.transactions:
            if txn["store_id"] != store_id:
                continue
            if start_time and txn["timestamp"] < start_time:
                continue
            if end_time and txn["timestamp"] > end_time:
                continue
            results.append(txn)
        return results
