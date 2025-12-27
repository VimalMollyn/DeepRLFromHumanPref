"""Human feedback API for RL-Teacher."""
import logging
import os

import django
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady


def initialize():
    """Initialize Django settings when importing this module standalone."""
    if settings.configured:
        return

    try:
        from human_feedback_site import settings as site_settings

        # Get all uppercase settings from the settings module
        settings_dict = {
            key: getattr(site_settings, key)
            for key in dir(site_settings)
            if key.isupper()
        }
        settings.configure(**settings_dict)
        django.setup()
    except RuntimeError:
        logging.warning(
            "Tried to double configure the API, ignore this if running the Django app directly"
        )


# Only initialize if we're being imported standalone (not via Django)
if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    initialize()

try:
    from human_feedback_api.models import Comparison
except AppRegistryNotReady:
    logging.info("Could not yet import Comparison - Django not ready")
    Comparison = None
