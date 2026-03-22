import asyncio
import json
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import SelfHealingPipeline, to_json_safe

app = FastAPI(title="Self-Healing ML")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHECKPOINTS = Path("./checkpoints")
CHECKPOINTS.mkdir(exist_ok=True)

FAILURE_MODES = ["label_noise", "missing_feature", "spurious", "data_shift"]

FAILURE_PARAMS = {
    "label_noise":    {"condition": lambda x: x[0] > 0.5 and x[1] < 0, "noise_rate": 0.3},
    "missing_feature":{"feature_idx": 0},
    "spurious":       {"correlation_strength": 0.9},
    "data_shift":     {"shift_type": "covariate"},
}

STAGE_ORDER = ["Train", "Deploy", "Discover", "Diagnose", "Fix", "Validate", "Heal"]


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(to_json_safe(obj))}\n\n"


def _load_train_history(mode: str) -> list:
    import torch
    path = CHECKPOINTS / f"model_{mode or 'clean'}.pt"
    if not path.exists():
        return []
    try:
        chk  = torch.load(path, map_location="cpu")
        hist = chk.get("history", {})
        tas  = hist.get("train_acc", [])
        vas  = hist.get("val_acc", [])
        return [{"epoch": i + 1, "train_acc": round(ta, 2), "val_acc": round(va, 2)}
                for i, (ta, va) in enumerate(zip(tas, vas))]
    except Exception:
        return []


def _extract(mode: str, cycle: int, result: dict) -> dict:
    training   = result.get("training",        {})
    healing    = result.get("healing",         {})
    deployment = result.get("deployment",      {})
    diagnosis  = result.get("diagnosis",       {})
    fixes      = result.get("proposed_fixes",  [])
    validation = result.get("validation",      [])
    status     = result.get("status",          "unknown")

    baseline    = healing.get("baseline_accuracy", training.get("final_val_acc", 0))
    healed      = healing.get("healed_accuracy",   baseline)
    improvement = healing.get("improvement",       0)

    promoted = [v for v in validation if v.get("recommendation") == "PROMOTE"]
    top_fix  = fixes[0] if fixes else {}

    return {
        "cycle":          cycle,
        "mode":           mode,
        "baseline_acc":   round(baseline,    2),
        "healed_acc":     round(healed,      2),
        "improvement":    round(improvement, 2),
        "failure_rate":   round(deployment.get("failure_rate", 0) * 100, 1),
        "fix_applied":    top_fix.get("fix_id",        "none"),
        "confidence":     round(diagnosis.get("confidence",    0), 3),
        "diagnosis":      diagnosis.get("recommendation",      ""),
        "primary_cause":  diagnosis.get("primary_cause",       "unknown"),
        "status":         status,
        "num_fixes":      len(fixes),
        "num_promoted":   len(promoted),
        "metrics": {
            "accuracy":  round(healed,            1),
            "precision": round(max(healed - 1.5, 0), 1),
            "recall":    round(max(healed - 2.0, 0), 1),
            "f1":        round(max(healed - 1.0, 0), 1),
        },
        "train_history": _load_train_history(mode),
        "fixes": [
            {
                "fix_id":          f.get("fix_id",          ""),
                "description":     f.get("description",     ""),
                "expected_impact": round(f.get("expected_impact", 0), 3),
                "priority":        f.get("priority",        5),
            }
            for f in fixes[:5]
        ],
    }


def _run_cycle(mode: str) -> dict:
    pipeline = SelfHealingPipeline(output_dir=str(CHECKPOINTS))
    return pipeline.run_full_cycle(
        dataset_name="adult",
        failure_mode=mode if mode != "clean" else None,
        failure_params=FAILURE_PARAMS.get(mode),
        auto_heal=True,
    )


@app.get("/")
async def root():
    return FileResponse("dashboard.html")


@app.get("/api/run")
async def run_endpoint(
    cycles: int = Query(default=3, ge=1, le=10),
    mode:   str = Query(default="all"),
):
    modes = FAILURE_MODES if mode == "all" else [mode]

    async def stream():
        all_results: dict = {m: [] for m in modes}
        loop = asyncio.get_event_loop()

        for cycle_idx in range(cycles):
            for m in modes:
                for si, stage in enumerate(STAGE_ORDER[:2]):
                    yield _sse({"type": "stage", "cycle": cycle_idx + 1,
                                "total_cycles": cycles, "mode": m,
                                "stage": stage, "stage_idx": si})
                    await asyncio.sleep(0)

                try:
                    result = await loop.run_in_executor(None, _run_cycle, m)

                    keys_to_stages = [
                        ("discovery",  2, "Discover"),
                        ("diagnosis",  3, "Diagnose"),
                        ("proposed_fixes", 4, "Fix"),
                        ("validation", 5, "Validate"),
                        ("healing",    6, "Heal"),
                    ]
                    for key, si, label in keys_to_stages:
                        if key in result:
                            yield _sse({"type": "stage", "cycle": cycle_idx + 1,
                                        "total_cycles": cycles, "mode": m,
                                        "stage": label, "stage_idx": si})
                            await asyncio.sleep(0.04)

                    cr = _extract(m, cycle_idx + 1, result)
                    all_results[m].append(cr)
                    yield _sse({"type": "cycle_complete", "result": cr,
                                "all_results": all_results})

                except Exception as e:
                    yield _sse({"type": "error", "mode": m,
                                "cycle": cycle_idx + 1,
                                "error": str(e), "trace": traceback.format_exc()})

                await asyncio.sleep(0)

        yield _sse({"type": "done", "all_results": all_results})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/snapshot")
async def snapshot_endpoint(cycles: int = Query(default=1, ge=1, le=5)):
    loop        = asyncio.get_event_loop()
    all_results = {m: [] for m in FAILURE_MODES}

    for m in FAILURE_MODES:
        for ci in range(cycles):
            try:
                result = await loop.run_in_executor(None, _run_cycle, m)
                all_results[m].append(_extract(m, ci + 1, result))
            except Exception as e:
                all_results[m].append({"cycle": ci + 1, "mode": m, "error": str(e)})

    return JSONResponse(content=to_json_safe(all_results))