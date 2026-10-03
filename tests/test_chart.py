"""The Helm chart renders settings homeduplex accepts. Needs `helm` (CI installs it); skipped without it."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from homeduplex.config import load

CHART = Path(__file__).parent.parent / "charts" / "homeduplex"


def _helm() -> str | None:
    """A helm that runs (an installed binary for the wrong CPU doesn't count)."""
    helm = os.environ.get("HELM") or shutil.which("helm")
    if helm is None:
        return None
    try:
        subprocess.run([helm, "version"], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return helm


HELM = _helm()

pytestmark = pytest.mark.skipif(HELM is None, reason="helm is not installed or does not run")


def render(*args: str) -> list[dict[str, Any]]:
    assert HELM is not None
    out = subprocess.run(
        [HELM, "template", "test", str(CHART), *args], check=True, capture_output=True, text=True
    ).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def by_kind(docs: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    [doc] = [d for d in docs if d["kind"] == kind]
    return doc


@pytest.mark.parametrize("values", sorted((CHART / "ci").glob("*-values.yaml")), ids=lambda p: p.stem)
def test_rendered_settings_load(values: Path, tmp_path: Path) -> None:
    docs = render("-f", str(values))
    settings_file = tmp_path / "homeduplex.yaml"
    settings_file.write_text(by_kind(docs, "ConfigMap")["data"]["homeduplex.yaml"])
    settings = load(settings_file, {})
    assert (settings.server.host, settings.server.port) == ("0.0.0.0", 8770)


def test_missing_backends_fail_with_a_clear_message() -> None:
    assert HELM is not None
    result = subprocess.run([HELM, "template", "test", str(CHART)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "config.stt is required" in result.stderr


def test_secure_defaults() -> None:
    docs = render("-f", str(CHART / "ci" / "minimal-values.yaml"))
    pod = by_kind(docs, "Deployment")["spec"]["template"]["spec"]
    [container] = pod["containers"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["automountServiceAccountToken"] is False
    assert pod["enableServiceLinks"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["readinessProbe"]["httpGet"]["path"] == "/healthz"
    assert not any(d["kind"] == "Ingress" for d in docs)
