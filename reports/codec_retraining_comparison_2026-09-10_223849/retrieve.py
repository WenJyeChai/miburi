"""Read-only W&B snapshot using the account's existing local authentication."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import netrc
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
ENTITY = "wenjye00-hong-kong-university-of-science-and-technology"
PROJECT = "miburi_single"


def query(document, variables):
    credentials = netrc.netrc(str(Path.home() / "_netrc")).authenticators("api.wandb.ai")
    if credentials is None:
        raise RuntimeError("Existing W&B authentication was not found.")
    response = requests.post(
        "https://api.wandb.ai/graphql",
        json={"query": document, "variables": variables},
        auth=("api", credentials[2]), timeout=60, allow_redirects=False,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("errors"):
        raise RuntimeError(result["errors"])
    return result["data"]["project"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", nargs="*")
    args = parser.parse_args()
    base = {"entity": ENTITY, "project": PROJECT}
    if args.histories is None:
        snapshot_at = datetime.now(timezone.utc).isoformat()
        result = query("""
          query Runs($entity:String!, $project:String!) {
            project(name:$project, entityName:$entity) {
              name runs(first:100, order:"-createdAt") {
                edges { node { name displayName state createdAt config summaryMetrics } }
                pageInfo { endCursor hasNextPage }
              }
            }
          }
        """, base)
        if result["runs"]["pageInfo"]["hasNextPage"]:
            raise RuntimeError("Run inventory needs another page.")
        runs = [edge["node"] for edge in result["runs"]["edges"]]
        (ROOT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
        metadata_path = ROOT / "retrieval.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        time_key = "summary_rechecked_at_utc" if "integrity" in metadata else "summary_snapshot_at_utc"
        metadata.update({time_key: snapshot_at, "entity": ENTITY, "project": PROJECT})
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for run in runs:
            summary = json.loads(run["summaryMetrics"])
            print(json.dumps({key: run[key] for key in ("name", "displayName", "state", "createdAt")}
                             | {"step": summary.get("_step"),
                                "epoch": summary.get("trainer/epoch"),
                                "val_ce": summary.get("val/ce_loss")}))
        return

    runs = json.loads((ROOT / "runs.json").read_text(encoding="utf-8"))
    selected = {run["name"]: run for run in runs if run["name"] in args.histories}
    assert set(selected) == set(args.histories)
    document = """
      query HistoryPage($entity:String!, $project:String!, $run:String!,
                        $minStep:Int64!, $maxStep:Int64!, $pageSize:Int!) {
        project(name:$project, entityName:$entity) {
          run(name:$run) {
            history(minStep:$minStep, maxStep:$maxStep, samples:$pageSize)
          }
        }
      }
    """
    jobs = []
    for run_id, run in selected.items():
        end = int(json.loads(run["summaryMetrics"])["_step"]) + 1
        jobs.extend((run_id, start, min(start + 500, end)) for start in range(0, end, 500))

    def page(job):
        run_id, start, end = job
        result = query(document, base | {"run": run_id, "minStep": start,
                                        "maxStep": end, "pageSize": 500})
        rows = [json.loads(row) if isinstance(row, str) else row
                for row in result["run"]["history"]]
        # The API fills absent metrics with null; retain only logged values.
        return run_id, [{key: value for key, value in row.items() if value is not None}
                        for row in rows]

    histories = {run_id: {} for run_id in selected}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for run_id, rows in pool.map(page, jobs):
            for row in rows:
                histories[run_id][row["_step"]] = row
    integrity = {}
    for run_id, indexed in histories.items():
        end = int(json.loads(selected[run_id]["summaryMetrics"])["_step"]) + 1
        missing = sorted(set(range(end)) - set(indexed))
        if missing:
            raise RuntimeError(f"{run_id}: {len(missing)} missing logged steps")
        rows = [indexed[step] for step in sorted(indexed)]
        (ROOT / f"{run_id}_history.json").write_text(json.dumps(rows), encoding="utf-8")
        integrity[run_id] = {"rows": len(rows), "first_step": min(indexed),
                             "last_step": max(indexed), "missing_steps": len(missing),
                             "validation_rows": sum("val/ce_loss" in row for row in rows)}
        print(json.dumps({"run_id": run_id} | integrity[run_id]))
    metadata = json.loads((ROOT / "retrieval.json").read_text(encoding="utf-8"))
    metadata.update({"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                     "history_sampling": "All logged steps, disjoint <=500-step ranges, 500 requested rows per range.",
                     "integrity": integrity})
    (ROOT / "retrieval.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
