import json
from argparse import Namespace

import pytest

from boxer.bin_state import BinState
from boxer.cli import bin_command, build_parser, controller_lock, main
from boxer.settings import BIN_IDS, SOURCE_IDS, Settings, SetupError


def test_teach_accepts_physical_source_and_bin_ids():
    for name in (*SOURCE_IDS, *BIN_IDS):
        assert build_parser().parse_args(["teach", name]).target == name


def test_bin_init_requires_explicit_empty_confirmation(settings, tmp_path):
    settings.data["state_file"] = str(tmp_path / "bins.json")
    with pytest.raises(SetupError, match="empty-confirmed"):
        bin_command(Namespace(bin_action="init", empty_confirmed=False), settings)
    assert not settings.state_path.exists()


def test_init_cannot_erase_existing_assignments_and_reset_backs_them_up(settings, tmp_path):
    settings.data["state_file"] = str(tmp_path / "bins.json")
    bin_command(Namespace(bin_action="init", empty_confirmed=True), settings)
    state = BinState.load(settings, settings.state_path)
    state.begin("yellow", "bin-1", "source-1")
    state.stage("release_pending")
    state.confirm_release()
    with pytest.raises(SetupError, match="already exists"):
        bin_command(Namespace(bin_action="init", empty_confirmed=True), settings)
    bin_command(Namespace(bin_action="reset", empty_confirmed=True), settings)
    assert BinState.load(settings, settings.state_path).data["bins"]["bin-1"]["color"] is None
    previous = settings.state_path.with_name(settings.state_path.name + ".previous.json")
    assert json.loads(previous.read_text())["bins"]["bin-1"]["color"] == "yellow"


def test_shared_state_file_lock_rejects_second_local_controller(tmp_path):
    path = tmp_path / "bins.json"
    with controller_lock(path), pytest.raises(SetupError, match="lock exists"), controller_lock(path):
        pytest.fail("Second controller acquired the same state file")
    assert not path.with_name(path.name + ".lock").exists()


def test_uncalibrated_execute_exits_before_connecting(tmp_path, monkeypatch):
    from boxer import cli

    settings = Settings.load(cli.ROOT / "config/sorter.example.json")
    path = tmp_path / "settings.json"
    settings.save(path)

    def forbidden(*args, **kwargs):
        pytest.fail("A connection was attempted with uncalibrated settings")

    monkeypatch.setattr(cli, "connected_command", forbidden)
    with pytest.raises(SystemExit) as result:
        main(["--config", str(path), "run", "--execute"])
    assert result.value.code == 1
