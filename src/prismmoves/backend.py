"""Supported PixelCue calls; no independent analysis or model engine."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


def validate_profile(profile):
    if profile.get("schema") != "pixelcue.visual-analysis/1":
        raise ValueError("Unsupported PixelCue analysis schema; PixelCue >=0.9.0 is required")
    for key in ("source", "measurements", "models", "limits"):
        if key not in profile:
            raise ValueError(f"Missing PixelCue profile field: {key}")
    return profile


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("PixelCue redirects are not accepted")


class PixelCueBackend:
    def __init__(self, url=None, timeout=300):
        self.url = url.rstrip("/") if url else None
        self.timeout = timeout
        self._service = None
        if self.url:
            parts = urllib.parse.urlsplit(self.url)
            if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1")
                or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/")):
                raise ValueError("PixelCue URL must be a loopback http origin, e.g. http://127.0.0.1:8765")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _request(self, route, body=None):
        request = urllib.request.Request(self.url+route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            content = response.read(16*1024**2+1)
            if len(content) > 16*1024**2:
                raise ValueError("PixelCue response exceeds 16 MiB")
            return json.loads(content)

    def identity(self):
        if self.url:
            health = self._request("/health")
            if health.get("service") != "pixelcue" or not health.get("visual_analysis"):
                raise ValueError("Selected service lacks PixelCue /v1/analysis; upgrade PixelCue to >=0.9.0")
            return {"mode": "pixelcue-http", "url": self.url, "health": health}
        import pixelcue
        from pixelcue.visual_service import VisualAnalysisService
        self._service = VisualAnalysisService()
        return {"mode": "pixelcue-python", "version": pixelcue.__version__}

    def analyse(self, path, **options):
        if self.url:
            result = self._request("/v1/analysis", {"path": str(path), **options})
        else:
            if self._service is None:
                self.identity()
            if options.get("keyword_models"):
                # Existing PixelCue keyword machinery remains the sole tagging implementation.
                from pixelcue.service import KeywordExtractionService
                self._service.keyword_service = KeywordExtractionService()
            result = self._service.analyse(path, **options)
        return validate_profile(result)
