"""Resumable local completion controller. It never starts paid compute services."""

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/v03"
REPORT = ROOT / "reports/v03"
CONTROL = RUN / "orchestration"
UV = os.environ.get("PATCHDEDUP_UV", str(Path.home() / ".local/bin/uv"))
HF = os.environ.get("PATCHDEDUP_HF_CLI", str(Path.home() / ".local/bin/hf"))
MODEL_ID = "enzoferracini/patchdedup-ptbr-mdeberta"
DATASET_ID = "enzoferracini/patchdedup-synthetic-ptbr"
CHILD = None
STAGES = [
    ("wait_existing", 5),
    ("restore_inputs", 3),
    ("train", 25),
    ("hpo", 12),
    ("adapt", 6),
    ("pseudo", 8),
    ("quantize", 3),
    ("search", 8),
    ("calibrate", 4),
    ("final_test", 4),
    ("render", 2),
    ("export", 3),
    ("checks", 5),
    ("docker", 3),
    ("publish", 4),
    ("verify_remote", 3),
    ("git", 2),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another completion controller already owns the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def log(message):
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def command(stage, arguments, env=None, timeout=6 * 3600):
    global CHILD
    folder = CONTROL / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / f"{stage}.log"
    merged = os.environ.copy()
    merged.update(PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUTF8="1")
    if env:
        merged.update(env)
    log(f"{stage}: log em {logfile}")
    with logfile.open("a", encoding="utf-8") as stream:
        stream.write("\n" + dt.datetime.now().isoformat() + " " + repr(arguments) + "\n")
        stream.flush()
        CHILD = subprocess.Popen(
            [str(a) for a in arguments],
            cwd=ROOT,
            env=merged,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result = CHILD.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGTERM)
            raise RuntimeError(f"{stage}: timeout; inspect {logfile}") from None
        finally:
            CHILD = None
    if result:
        raise RuntimeError(f"{stage}: exit code {result}; inspect {logfile}")


def python(stage, script, *args, env=None):
    command(stage, [UV, "run", "--locked", "--extra", "ml", "python", script, *args], env=env)


def research_workers():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [
                p.decode(errors="replace")
                for p in (entry / "cmdline").read_bytes().split(b"\0")
                if p
            ]
            cwd = (entry / "cwd").resolve()
        except (OSError, RuntimeError):
            continue
        if not args or not Path(args[0]).name.startswith("python") or cwd != ROOT:
            continue
        if any(Path(a).name in ("study_v03.py", "extra_v03.py") for a in args[1:3]):
            found.append(int(entry.name))
    return found


def wait_existing():
    started = time.monotonic()
    idle = 0
    while idle < 3:
        workers = research_workers()
        if workers:
            idle = 0
            log(f"Aguardando trabalhadores existentes: {workers}")
            write(CONTROL / "waiting.json", dict(pids=workers, waiting=True))
            time.sleep(10)
        else:
            idle += 1
            time.sleep(2)
        if time.monotonic() - started > 6 * 3600:
            raise RuntimeError(
                "Existing workers exceeded six hours; no duplicate worker was started"
            )
    write(CONTROL / "waiting.json", dict(pids=[], waiting=False))


def archive_partial(directory, completion_file):
    directory = Path(directory)
    if directory.exists() and not (directory / completion_file).exists():
        destination = RUN / "aborted" / f"{directory.name}-{time.time_ns()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(directory), str(destination))
        log(f"Checkpoint incompleto preservado em {destination}")
        return destination
    return None


def restore_inputs():
    cfg = read(ROOT / "configs/experiment-v03.json")
    for item in cfg["nli_models"].values():
        target = ROOT / item["directory"]
        if not (target / "model.safetensors").exists():
            # Only root checkpoint/tokenizer files; exclude unrelated ONNX variants.
            from huggingface_hub import HfApi

            info = HfApi().model_info(item["repo"], revision=item["revision"])
            names = [
                s.rfilename
                for s in info.siblings
                if "/" not in s.rfilename
                and (
                    s.rfilename.endswith(".json")
                    or s.rfilename
                    in ("model.safetensors", "spm.model", "sentencepiece.bpe.model", "README.md")
                )
            ]
            command(
                "download-" + target.name,
                [
                    HF,
                    "download",
                    item["repo"],
                    *names,
                    "--revision",
                    item["revision"],
                    "--local-dir",
                    str(target),
                ],
            )
    if (
        not (RUN / "cache/nli_deberta-development.npy").exists()
        or not (RUN / "cache/nli_minilm-development.npy").exists()
    ):
        python("baseline", "scripts/study_v03.py", "baseline")
    release = read(ROOT / "configs/release.json")
    old = release.get("history", {}).get("v02", release)
    if not (ROOT / "artifacts/extended-42/selected-model/manifest.json").exists():
        command(
            "restore-embedding",
            [
                HF,
                "download",
                "enzoferracini/patchdedup-ptbr-minilm",
                "--revision",
                old["model_revision"],
                "--local-dir",
                "artifacts/extended-42/selected-model",
            ],
        )
    if not (ROOT / "artifacts/publication/dataset/README.md").exists():
        command(
            "restore-dataset",
            [
                HF,
                "download",
                DATASET_ID,
                "--type",
                "dataset",
                "--revision",
                old["dataset_revision"],
                "--local-dir",
                "artifacts/publication/dataset",
            ],
        )


def train_complete():
    names = [
        "multitask_minilm",
        "multitask_deberta",
        "lora_deberta",
        "distilled_minilm",
        "seed137_minilm",
    ]
    return (
        all(
            (REPORT / "candidates" / f"{name}.json").exists()
            and (RUN / name / "training.json").exists()
            for name in names
        )
        and (REPORT / "teacher.json").exists()
    )


def finish_train():
    if train_complete():
        return
    for name in [
        "multitask_minilm",
        "multitask_deberta",
        "lora_deberta",
        "distilled_minilm",
        "seed137_minilm",
    ]:
        archive_partial(RUN / name, "training.json")
    python("train", "scripts/study_v03.py", "train")
    if not train_complete():
        raise RuntimeError("Training stage did not produce all required checkpoints")


def finish_hpo():
    import optuna

    storage = f"sqlite:///{RUN / 'search.sqlite3'}"
    if (RUN / "search.sqlite3").exists():
        study = optuna.load_study(study_name="patchdedup-v03", storage=storage)
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                archive_partial(RUN / f"hpo_{trial.number}", "training.json")
    target = read(ROOT / "configs/experiment-v03.json")["hpo_trials"]
    for _ in range(3):
        if (REPORT / "hpo.json").exists():
            done = sum(
                t["state"] in ("COMPLETE", "PRUNED") for t in read(REPORT / "hpo.json")["trials"]
            )
            if done >= target:
                return
        python("hpo", "scripts/study_v03.py", "hpo")
    raise RuntimeError("HPO has insufficient successful/pruned trials after retries")


def extra(stage):
    if stage == "adapt":
        if (REPORT / "candidates/domain_adapted_minilm.json").exists():
            return
        archive_partial(RUN / "language_adaptation", "adaptation.json")
        archive_partial(RUN / "domain_adapted_minilm", "training.json")
    elif stage == "pseudo":
        if (REPORT / "pseudo-labels.json").exists():
            accepted = read(REPORT / "pseudo-labels.json")["accepted"]
            if not accepted or (REPORT / "candidates/pseudo_distilled_minilm.json").exists():
                return
        archive_partial(RUN / "pseudo_distilled_minilm", "training.json")
    elif (REPORT / "quantization.json").exists():
        return
    try:
        python(stage, "scripts/extra_v03.py", stage)
    except RuntimeError as exc:
        if stage != "quantize":
            raise
        write(
            REPORT / "quantization.json",
            dict(
                status="unsupported_or_failed",
                error=str(exc),
                note="Float models remain candidates; no unsupported quantized model was selected",
            ),
        )
        log("Quantização registrada como falha; continuando com modelos float verificados.")


def checked_export():
    from patchdedup.encoder import verify_bundle

    model = RUN / "publication/model"
    if (RUN / "publication").exists():
        try:
            verify_bundle(model)
            if (RUN / "publication/dataset/v03/test.jsonl").exists():
                return
        except (OSError, ValueError):
            pass
        source = RUN / "publication"
        destination = RUN / "aborted" / f"publication-{time.time_ns()}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    python("export", "scripts/export_v03.py")


def check_local():
    command("lint", [UV, "run", "--locked", "--extra", "ml", "ruff", "check", "."])
    command("format", [UV, "run", "--locked", "--extra", "ml", "ruff", "format", "--check", "."])
    command(
        "tests",
        [UV, "run", "--locked", "--extra", "ml", "pytest", "-q"],
        env={
            "PATCHDEDUP_MODEL": str(RUN / "publication/model"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    command("build", [UV, "build"])


def publish():
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    info = None
    try:
        info = HfApi().model_info(MODEL_ID)
    except RepositoryNotFoundError:
        command("create-model", [HF, "repos", "create", MODEL_ID, "--type", "model", "--public"])
    if info is not None:
        files = {p.rfilename for p in info.siblings}
        if "patchdedup.json" not in files and files - {".gitattributes", "README.md"}:
            raise RuntimeError("Destination contains an unknown model; refusing overwrite")
    command(
        "upload-model",
        [
            HF,
            "upload",
            MODEL_ID,
            str(RUN / "publication/model"),
            ".",
            "--commit-message",
            "feat(model): publish selected local mDeBERTa research pipeline",
        ],
    )
    command(
        "upload-dataset",
        [
            HF,
            "upload",
            DATASET_ID,
            str(RUN / "publication/dataset"),
            ".",
            "--type",
            "dataset",
            "--commit-message",
            "feat(data): publish grouped v03 development calibration and test data",
        ],
    )
    api = HfApi()
    model = api.model_info(MODEL_ID)
    dataset = api.dataset_info(DATASET_ID)
    old = read(ROOT / "configs/release.json")
    history = old.get("history", {})
    if old["model_id"] != MODEL_ID:
        history["v02"] = old
    write(
        ROOT / "configs/release.json",
        dict(
            status="experimental-v03",
            model_id=MODEL_ID,
            model_revision=model.sha,
            dataset_id=DATASET_ID,
            dataset_revision=dataset.sha,
            history=history,
        ),
    )
    write(CONTROL / "published.json", dict(model_revision=model.sha, dataset_revision=dataset.sha))


def verify_remote():
    from patchdedup.encoder import Comparator, verify_bundle
    from patchdedup.schemas import CompareInput

    release = read(ROOT / "configs/release.json")
    directory = RUN / "downloaded-model"
    command(
        "download-published",
        [
            HF,
            "download",
            MODEL_ID,
            "--revision",
            release["model_revision"],
            "--local-dir",
            str(directory),
        ],
    )
    verify_bundle(directory)
    result = Comparator(directory).compare(
        CompareInput(
            text_a="Automatizei relatórios com Python.",
            text_b="Usei Python para automatizar relatórios.",
        )
    )
    write(REPORT / "published-inference.json", result.model_dump())
    command(
        "published-smoke",
        [
            UV,
            "run",
            "--locked",
            "--extra",
            "ml",
            "pytest",
            "tests/test_encoder.py",
            "-m",
            "real_model",
            "-q",
        ],
        env={
            "PATCHDEDUP_MODEL": str(directory),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )


def git_publish():
    paths = [
        "src",
        "scripts",
        "configs",
        "data/v03",
        "data/raw/development-v03.psv",
        "docs",
        "reports/v03",
        "hub",
        "licenses",
        "README.md",
        "CONTRIBUTING.md",
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
        "tests",
        ".github",
    ]
    command("git-add", ["git", "add", "--", *paths])
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False)
    if result.returncode == 1:
        command(
            "git-commit",
            ["git", "commit", "-m", "feat(release): publish evaluated mDeBERTa research pipeline"],
        )
    elif result.returncode:
        raise RuntimeError("Unable to inspect staged changes")
    command("git-push", ["git", "push"])


def stop(signum, _frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    log("Executor interrompido; checkpoints concluídos foram preservados.")
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload verified artifacts and push generated research changes",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without starting jobs or writing state"
    )
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    stages=[s for s, _ in STAGES],
                    publish=args.publish,
                    existing_workers=research_workers(),
                    paid_services=False,
                ),
                indent=2,
            )
        )
        return
    os.chdir(ROOT)
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError(
            "At least 20 GiB of free disk is required for checkpoints and verification"
        )
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    CONTROL.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(CONTROL / "run.lock"):
        (CONTROL / "pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
        if (CONTROL / "status.json").exists():
            prior = read(CONTROL / "status.json")
            if prior.get("status") == "complete" and (not args.publish or prior.get("published")):
                log("Todas as etapas solicitadas já foram concluídas.")
                return
        actions = dict(
            wait_existing=wait_existing,
            restore_inputs=restore_inputs,
            train=finish_train,
            hpo=finish_hpo,
            adapt=lambda: extra("adapt"),
            pseudo=lambda: extra("pseudo"),
            quantize=lambda: extra("quantize"),
            search=lambda: None
            if (REPORT / "selection.json").exists()
            else python("search", "scripts/search_scores_v03.py"),
            calibrate=lambda: None
            if (REPORT / "calibration.json").exists()
            else python("calibrate", "scripts/calibrate_v03.py"),
            final_test=lambda: None
            if (REPORT / "test.json").exists()
            else python("final-test", "scripts/evaluate_final_v03.py"),
            render=lambda: python("render", "scripts/render_v03.py"),
            export=checked_export,
            checks=check_local,
            docker=lambda: command(
                "docker", ["bash", "scripts/docker_smoke.sh", str(RUN / "publication/model")]
            ),
            publish=publish if args.publish else lambda: None,
            verify_remote=verify_remote if args.publish else lambda: None,
            git=git_publish if args.publish else lambda: None,
        )
        finished = 0
        total = sum(weight for _, weight in STAGES)
        for stage, weight in STAGES:
            status = dict(
                status="running",
                stage=stage,
                percent=round(100 * finished / total),
                pid=os.getpid(),
                published=args.publish,
                updated=dt.datetime.now().isoformat(),
            )
            write(CONTROL / "status.json", status)
            log(f"{stage}: {status['percent']}% da execução do controlador")
            try:
                actions[stage]()
            except Exception as exc:
                status.update(status="failed", error=str(exc))
                write(CONTROL / "status.json", status)
                log(str(exc))
                raise
            finished += weight
        write(
            CONTROL / "status.json",
            dict(
                status="complete",
                percent=100,
                published=args.publish,
                model=MODEL_ID if args.publish else str(RUN / "publication/model"),
                updated=dt.datetime.now().isoformat(),
            ),
        )
        log("Concluído. Relatório: reports/v03/README.md")


if __name__ == "__main__":
    main()
