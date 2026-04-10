"""External API tools — vendor lookup and any future third-party integrations.

Currently provides:
- ``lookup_vendor_external``: Placeholder for future vendor registry API calls.
  Returns a stub response so the rest of the agent system works without
  needing live credentials during development.

Add real integrations here without touching any agent or tool logic.
"""

import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def lookup_vendor_external(vendor_name: str) -> str:
    """Look up a vendor in an external registry (e.g. tax authority, credit bureau).

    Currently returns a stub response. Replace the body of this function
    with real API calls when credentials are available.

    Args:
        vendor_name: Vendor name to look up.

    Returns:
        JSON string with vendor registration status.
    """
    logger.info("lookup_vendor_external (stub) called for vendor='%s'.", vendor_name)
    # TODO: Replace with real external API call, e.g.:
    #   response = httpx.get(f"https://vendor-registry.example.com/api/v1/lookup?name={vendor_name}")
    #   return response.text
    return json.dumps({
        "vendor": vendor_name,
        "source": "external_registry_stub",
        "registered": True,
        "registration_number": "STUB-000",
        "country": "TN",
        "status": "active",
        "note": "Stub response — replace with real API call.",
    })
