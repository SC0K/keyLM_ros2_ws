"""Wire conventions for preview-only goals and explicit controller approval."""

# An unconfigured controller cannot accidentally execute preview traffic.
PREVIEW_SUFFIX = "/preview"
APPROVED_SUFFIX = "/approved"
CANCEL_SUFFIX = "/cancel_preview"
SET_MODE_SUFFIX = "/set_supervised_mode"
PREVIEW_LEASE_SEC = 3.0
PREVIEW_REFRESH_SEC = 0.5


def preview_id(message):
    return message.layout.dim[0].label if message.layout.dim else ""
