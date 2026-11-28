"""Private signed standing offers, accepted before an order opens."""

from __future__ import annotations

import asyncio
import sqlite3
import time

from fastapi import APIRouter, HTTPException

from reliquary.protocol.corpus_submission import StandingOfferRequest
from reliquary.protocol.signatures import verify_corpus_lease_signature


def standing_offers_router(
    observation,
    *,
    registration,
    operator_of,
    verify=verify_corpus_lease_signature,
    clock=time.time,
):
    router = APIRouter()

    @router.put("/corpus/standing-offer")
    async def put_offer(body: StandingOfferRequest) -> dict:
        if body.market not in observation.markets:
            raise HTTPException(404, "market_not_served")
        if (body.slots == 0) != (body.tokens_per_second == 0):
            raise HTTPException(422, "withdraw_with_zero_rate_and_slots")
        if not verify(body):
            raise HTTPException(403, "bad_signature")
        if registration is None or operator_of is None:
            raise HTTPException(503, "registration_unavailable")
        reason = await registration(body.miner_hotkey)
        if reason:
            raise HTTPException(403 if reason == "not_registered" else 503, reason)
        owner = operator_of(body.miner_hotkey)
        if not owner:
            raise HTTPException(503, "owner_unavailable")
        if owner != body.coldkey:
            raise HTTPException(403, "owner_changed")
        def record():
            observation.book.post_offer(
                body.market, body.miner_hotkey, owner, price_micro=body.micro_alpha_per_mtoken,
                rate=body.tokens_per_second, slots=body.slots, sequence=body.sequence, at=int(clock()),
            )

        try:
            await asyncio.to_thread(record)
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(503, "observation_unavailable") from exc
        except ValueError as exc:
            raise HTTPException(409, "offer_conflict") from exc
        return {"accepted": True, "sequence": body.sequence, "mode": "observation"}

    return router
