"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from pymem.config.settings import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "pymem",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
    )
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        beat_schedule={
            "nightly-consolidation": {
                "task": "pymem.tasks.consolidation_task.nightly_consolidation",
                "schedule": 86400,  # daily
            },
            "decay-check": {
                "task": "pymem.tasks.decay_task.run_decay",
                "schedule": 86400,  # daily
            },
        },
    )
    return app


celery_app = create_celery_app()
