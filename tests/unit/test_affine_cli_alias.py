"""The normal CLI delegates options to the separate native bridge."""
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from reliquary.cli.main import app


class AffineAlias(unittest.TestCase):
    def test_alias_preserves_options_exit_status_and_native_help(self):
        arguments = ["prepare", "--config", "/private/input", "--out", "/private/prepared"]
        with patch("reliquary.integrations.affine_cli.main", return_value=1) as native:
            result = CliRunner().invoke(app, ["affine", *arguments])
        self.assertEqual(result.exit_code, 1)
        native.assert_called_once_with(arguments)
        result = CliRunner().invoke(app, ["affine", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("prepare", result.output)
        self.assertIn("--out", result.output)


if __name__ == "__main__":
    unittest.main()
