import uuid

import pytest

from utils import *


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.slot_checkpoint_max_count = 2
    server.slot_checkpoint_triggers = "explicit"
    server.temperature = 0.0


def save_checkpoint(slot: int, session_id: str, name: str):
    return server.make_request(
        "POST",
        f"/slots/{slot}?action=checkpoint",
        data={"session_id": session_id, "name": name},
    )


def test_named_checkpoint_limit_replace_list_and_restore():
    global server
    session_id = f"test-{uuid.uuid4().hex}"
    server.start()

    completion = server.make_request("POST", "/completion", data={
        "prompt": "A stable system prompt.",
        "id_slot": 1,
        "n_predict": 1,
        "cache_prompt": True,
    })
    assert completion.status_code == 200

    system_checkpoint = save_checkpoint(1, session_id, "system_prompt_end")
    assert system_checkpoint.status_code == 200
    assert system_checkpoint.body["checkpoint"]["name"] == "system_prompt_end"
    assert system_checkpoint.body["checkpoint"]["valid"] is True

    latest_checkpoint = save_checkpoint(1, session_id, "session_latest")
    assert latest_checkpoint.status_code == 200

    replaced_checkpoint = save_checkpoint(1, session_id, "session_latest")
    assert replaced_checkpoint.status_code == 200
    assert replaced_checkpoint.body["checkpoint"]["created_at"] <= replaced_checkpoint.body["checkpoint"]["updated_at"]

    listed = server.make_request(
        "GET",
        f"/slots/1/checkpoints?session_id={session_id}",
    )
    assert listed.status_code == 200
    assert listed.body["max_count"] == 2
    assert sorted(item["name"] for item in listed.body["data"]) == [
        "session_latest",
        "system_prompt_end",
    ]

    rejected = save_checkpoint(1, session_id, "third_checkpoint")
    assert rejected.status_code >= 400

    restored = server.make_request(
        "POST",
        "/slots/0?action=checkpoint_restore",
        data={"session_id": session_id, "name": "system_prompt_end"},
    )
    assert restored.status_code == 200
    assert restored.body["checkpoint"]["name"] == "system_prompt_end"
    assert restored.body["operation"]["n_restored"] > 0
