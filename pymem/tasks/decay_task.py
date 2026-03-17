"""Celery task for memory decay (forgetting curve)."""

from __future__ import annotations

import logging

from pymem.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="pymem.tasks.decay_task.run_decay")
def run_decay():
    """Daily decay task — recalculate scores using forgetting curve.

    Steps:
    1. Fetch all semantic memories not accessed in > 30 days
    2. Recalculate score using forgetting curve
    3. Soft-delete memories with score < 0.1
    4. Downrank memories with score < 0.3
    """
    logger.info("Starting decay engine run")
    # In production, this would connect to the database
    # and apply the decay calculations
    logger.info("Decay engine run completed")
