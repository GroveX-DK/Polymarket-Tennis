"""Live order execution on the Polymarket CLOB (only imported when TRADE_MODE=live).

Uses the official py-clob-client (pip install py-clob-client). Credentials come
from environment variables loaded out of pi_bot/.env - see .env.example.

  POLYMARKET_PRIVATE_KEY   Polygon wallet private key that signs orders
  POLYMARKET_FUNDER        Polymarket proxy/deposit address holding the USDC
                           (shown on the Polymarket deposit page). Leave empty
                           when trading straight from the key's own address.
  POLYMARKET_SIGNATURE_TYPE  1 = email/Magic login, 2 = browser-wallet proxy
                           (default when FUNDER is set), 0 = raw EOA.
"""
import os

CLOB_HOST = "https://clob.polymarket.com"
POLYGON = 137


class LiveTrader:
    def __init__(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        self._OrderArgs, self._OrderType, self._BUY = OrderArgs, OrderType, BUY

        pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "").strip()
        if not pk:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY not set")
        funder = os.environ.get("POLYMARKET_FUNDER", "").strip()
        sig = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
        kwargs = {"key": pk, "chain_id": POLYGON}
        if funder:
            kwargs["funder"] = funder
            kwargs["signature_type"] = int(sig) if sig else 2
        elif sig:
            kwargs["signature_type"] = int(sig)
        self.client = ClobClient(CLOB_HOST, **kwargs)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        self.address = self.client.get_address()

    def buy(self, token_id, price, usdc):
        """Marketable GTC limit buy: `usdc` worth of `token_id` at `price`.
        Returns the exchange order id. Raises on rejection."""
        price = round(float(price), 2)
        size = round(usdc / price, 2)
        if size * price < 1.0:
            raise ValueError(f"order cost ${size * price:.2f} below $1 minimum")
        order = self.client.create_order(self._OrderArgs(
            token_id=str(token_id), price=price, size=size, side=self._BUY))
        resp = self.client.post_order(order, self._OrderType.GTC)
        if not isinstance(resp, dict) or not resp.get("success"):
            raise RuntimeError(f"order rejected: {resp}")
        return resp.get("orderID", "")
