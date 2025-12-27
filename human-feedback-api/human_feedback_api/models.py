"""Django models for the human feedback API."""
from django.core.exceptions import ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _

RESPONSE_KIND_TO_RESPONSES_OPTIONS = {"left_or_right": ["left", "right", "tie", "abstain"]}


def validate_inclusion_of_response_kind(value):
    """Validate that response_kind is one of the allowed values."""
    kinds = RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys()
    if value not in kinds:
        raise ValidationError(
            _("%(value)s is not included in %(kinds)s"),
            params={"value": value, "kinds": list(kinds)},
        )


class Comparison(models.Model):
    """A comparison between two trajectory segments for human feedback."""

    created_at = models.DateTimeField("date created", auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    media_url_1 = models.TextField("media url #1", db_index=True)
    media_url_2 = models.TextField("media url #2", db_index=True)

    shown_to_tasker_at = models.DateTimeField(
        "time shown to tasker", db_index=True, blank=True, null=True
    )
    responded_at = models.DateTimeField(
        "time response received", db_index=True, blank=True, null=True
    )
    response_kind = models.TextField(
        "the type of response expected",
        db_index=True,
        validators=[validate_inclusion_of_response_kind],
    )
    response = models.TextField(
        "the response from the tasker", db_index=True, blank=True, null=True
    )
    experiment_name = models.TextField("name of experiment")

    priority = models.FloatField(
        "site will display higher priority items first", db_index=True
    )
    note = models.TextField(
        "note to be displayed along with the query", default="", blank=True
    )

    class Meta:
        ordering = ["-priority", "-created_at"]

    def __str__(self):
        return f"Comparison {self.id} ({self.experiment_name})"

    def full_clean(self, exclude=None, validate_unique=True):
        """Validate the model instance."""
        super().full_clean(exclude=exclude, validate_unique=validate_unique)
        self.validate_inclusion_of_response()

    @property
    def response_options(self):
        """Get the valid response options for this comparison's response_kind."""
        try:
            return RESPONSE_KIND_TO_RESPONSES_OPTIONS[self.response_kind]
        except KeyError:
            raise KeyError(
                f"{self.response_kind} is not a valid response_kind. "
                f"Valid response_kinds are {list(RESPONSE_KIND_TO_RESPONSES_OPTIONS.keys())}"
            )

    def validate_inclusion_of_response(self):
        """Validate that the response is one of the allowed options."""
        if self.response is not None and self.response not in self.response_options:
            raise ValidationError(
                _("%(value)s is not included in %(options)s"),
                params={"value": self.response, "options": self.response_options},
            )
