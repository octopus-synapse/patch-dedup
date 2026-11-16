import json
import subprocess
import sys


def run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "patchdedup", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_data_outside_repository(records, write_data, tmp_path):
    result = run_cli("validate-data", "--input", str(write_data(records)), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"]
    assert result.stderr == ""


def test_invalid_data_exit_code(write_data, tmp_path):
    result = run_cli("validate-data", "--input", str(write_data([])), cwd=tmp_path)
    assert result.returncode == 1
    assert json.loads(result.stdout)["valid"] is False


def test_missing_file_is_safe_json(tmp_path):
    result = run_cli("validate-data", "--input", "private-file.jsonl", cwd=tmp_path)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "INPUT_UNAVAILABLE"
    assert "private-file" not in result.stderr
    assert "Traceback" not in result.stderr


def test_usage_error_is_json(tmp_path):
    result = run_cli("validate-data", cwd=tmp_path)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "USAGE_ERROR"


def test_version(tmp_path):
    result = run_cli("--version", cwd=tmp_path)
    assert result.returncode == 0
    assert "0.1.0.dev0" in result.stdout


def test_installed_console_entrypoint(tmp_path):
    result = subprocess.run(
        [str(__import__("pathlib").Path(sys.executable).parent / "patchdedup"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "validate-data" in result.stdout


def test_compare_requires_explicit_arguments(tmp_path):
    result = run_cli("compare", cwd=tmp_path)
    assert result.returncode == 2


def test_compare_missing_bundle(tmp_path):
    result = run_cli(
        "compare",
        "--model",
        str(tmp_path / "absent"),
        "--text-a",
        "A",
        "--text-b",
        "B",
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "MODEL_UNAVAILABLE"


def test_compare_empty_text_is_safe(tmp_path):
    result = run_cli(
        "compare", "--model", str(tmp_path), "--text-a", " ", "--text-b", "B", cwd=tmp_path
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "EMPTY_TEXT"
