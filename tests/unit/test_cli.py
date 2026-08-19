"""Tests for Click CLI application."""

from __future__ import annotations

import json
from click.testing import CliRunner

from physics_reasoning.cli.main import cli


class TestCLI:
    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_validate_kb(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["validate-kb"])
        assert result.exit_code == 0
        assert "validation PASSED" in result.output
