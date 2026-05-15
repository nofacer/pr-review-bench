#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_VERSION = "docker-score-v1"
DEFAULT_MODEL = "gpt-5.5"


@dataclass
class Issue:
    title: str | None = None
    description: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    rule_name: str | None = None
    problematic_code_snippet: str | None = None


@dataclass
class MatchResult:
    matched_pred_indices: list[int]
    matched_truth_indices: list[int]
    reasons: list[str]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_issue(raw: dict[str, Any]) -> Issue:
    start_line = parse_int(raw.get("start_line"))
    end_line = parse_int(raw.get("end_line"))
    if start_line is not None and end_line is not None and start_line > end_line:
        start_line, end_line = end_line, start_line

    def text(name: str) -> str | None:
        value = raw.get(name)
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    return Issue(
        title=text("title"),
        description=text("description"),
        file_path=text("file_path"),
        start_line=start_line,
        end_line=end_line,
        rule_name=text("rule_name"),
        problematic_code_snippet=text("problematic_code_snippet"),
    )


def issue_fingerprint(issue: Issue) -> dict[str, Any]:
    return {
        "title": issue.title,
        "description": issue.description,
        "file_path": issue.file_path,
        "start_line": issue.start_line,
        "end_line": issue.end_line,
        "rule_name": issue.rule_name,
    }


def cache_key(model: str, case_id: str, truths: list[Issue], preds: list[Issue]) -> str:
    payload = {
        "version": PROMPT_VERSION,
        "model": model,
        "case_id": case_id,
        "truths": [issue_fingerprint(issue) for issue in truths],
        "predictions": [issue_fingerprint(issue) for issue in preds],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def format_issues(issues: list[Issue], label: str) -> str:
    if not issues:
        return "(none)"

    lines = []
    for index, issue in enumerate(issues):
        title = issue.title or "(no title)"
        desc = issue.description or "(no description)"
        file_path = issue.file_path or "(unknown)"
        start = issue.start_line if issue.start_line is not None else "?"
        end = issue.end_line if issue.end_line is not None else "?"
        rule = f" rule={issue.rule_name}" if issue.rule_name else ""
        snippet = issue.problematic_code_snippet
        snippet_line = f"\n    snippet: {snippet[:300]}" if snippet else ""
        lines.append(
            f"[{label} #{index}] file={file_path} lines={start}-{end}{rule}\n"
            f"    title: {title}\n"
            f"    description: {desc}"
            f"{snippet_line}"
        )
    return "\n".join(lines)


def build_prompt(batch: list[tuple[str, str, list[Issue], list[Issue]]]) -> str:
    matching_rules = (
        "## Matching rules (a prediction is a HIT for a ground-truth issue only if ALL apply)\n"
        "1. Same underlying issue: the prediction describes the same concrete defect, "
        "root cause, or policy violation as the ground-truth issue. Different wording "
        "is fine; a different problem, even if nearby, is not a match.\n"
        "2. Correct localization: the prediction points to the same file, AND the "
        "line range either overlaps the ground-truth range or is within ~5 lines of it.\n"
        "3. One-to-one: each prediction matches at most one ground-truth issue, and "
        "each ground-truth issue is matched by at most one prediction. If several "
        "predictions could match the same ground-truth issue, pick the single best "
        "match and treat the others as unmatched."
    )

    case_blocks = []
    case_ids = []
    for case_id, repo, truths, preds in batch:
        case_ids.append(case_id)
        case_blocks.append(
            f"## Case {case_id} (repo: {repo})\n"
            f"### Ground-truth issues\n{format_issues(truths, 'GT')}\n\n"
            f"### Predicted issues\n{format_issues(preds, 'PRED')}"
        )

    return (
        "You are an evaluator for an AI code review benchmark. For each case below, "
        "compare the predicted review findings against the ground-truth issues and "
        "decide which predictions correctly identified which real issues. Judge each "
        "case independently; indices are scoped per case.\n\n"
        f"{matching_rules}\n\n"
        f"{chr(10).join(case_blocks)}\n\n"
        "## Output\n"
        "Return ONLY a JSON object with this exact shape, and nothing else:\n"
        "{\n"
        '  "cases": {\n'
        '    "<case_id>": {\n'
        '      "matches": [\n'
        '        {"pred_idx": <int>, "gt_idx": <int>, "reason": "<short reason, <=120 chars>"}\n'
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n\n"
        f"- Include an entry for every case in this request: {', '.join(case_ids)}.\n"
        "- Use the 0-based indices shown as [GT #i] / [PRED #i] within each case.\n"
        "- Include only valid pairs that satisfy ALL matching rules.\n"
        "- Do not include unmatched items; the scorer derives FN and FP from what is missing.\n"
        "- No prose, no markdown fences, just the JSON object."
    )


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace == -1:
            raise RuntimeError(f"Codex response has no JSON object:\n{raw[:500]}")
        text = text[brace:]

    depth = 0
    end = None
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is not None:
        text = text[:end]
    return json.loads(text)


def call_codex(prompt: str, model: str, timeout: int) -> str:
    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    args = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-rules",
        "--model",
        model,
        "--cd",
        str(Path.cwd()),
        "--skip-git-repo-check",
    ]

    output_path = None
    raw_output = ""
    try:
        with tempfile.NamedTemporaryFile(
            prefix="benchmark-judge-",
            suffix=".txt",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)
        args.extend(["--output-last-message", str(output_path), prompt])
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    finally:
        if output_path is not None and output_path.exists():
            raw_output = output_path.read_text(encoding="utf-8", errors="replace")
            output_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"codex exited with code {proc.returncode}:\n{tail}")

    return raw_output or proc.stdout


def parse_matches(matches: Any, num_preds: int, num_truths: int) -> MatchResult:
    if not isinstance(matches, list):
        return MatchResult([], [], [])

    matched_preds = []
    matched_truths = []
    reasons = []
    seen_preds = set()
    seen_truths = set()

    for item in matches:
        if not isinstance(item, dict):
            continue
        try:
            pred_idx = int(item["pred_idx"])
            truth_idx = int(item["gt_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= pred_idx < num_preds):
            continue
        if not (0 <= truth_idx < num_truths):
            continue
        if pred_idx in seen_preds or truth_idx in seen_truths:
            continue
        seen_preds.add(pred_idx)
        seen_truths.add(truth_idx)
        matched_preds.append(pred_idx)
        matched_truths.append(truth_idx)
        reasons.append(str(item.get("reason", ""))[:200])

    return MatchResult(matched_preds, matched_truths, reasons)


def compute_score(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate(case_scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_tp = sum(score["tp"] for score in case_scores.values())
    total_fp = sum(score["fp"] for score in case_scores.values())
    total_fn = sum(score["fn"] for score in case_scores.values())
    metrics = compute_score(total_tp, total_fp, total_fn)
    metrics["cases"] = len(case_scores)
    metrics["total_tp"] = total_tp
    metrics["total_fp"] = total_fp
    metrics["total_fn"] = total_fn
    return metrics


def load_cases(dataset_json: Path) -> dict[str, dict[str, Any]]:
    data = load_json(dataset_json)
    if not isinstance(data, dict):
        raise RuntimeError(f"dataset.json must be an object: {dataset_json}")
    return data


def result_files(results_dir: Path) -> list[Path]:
    skip = {"benchmark_scope", "scores"}
    return sorted(
        path for path in results_dir.glob("*.json")
        if path.stem not in skip
    )


def read_predictions(path: Path) -> list[Issue]:
    try:
        data = load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"warn {path.name}: failed to parse result JSON, treating as no detections: {exc}",
            file=sys.stderr,
        )
        return []
    issues = data.get("issues", []) if isinstance(data, dict) else []
    if not isinstance(issues, list):
        issues = []
    return [parse_issue(item) for item in issues if isinstance(item, dict)]


def read_truths(case_data: dict[str, Any]) -> list[Issue]:
    issues = case_data.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return [parse_issue(item) for item in issues if isinstance(item, dict)]


def read_cache(cache_dir: Path | None, case_id: str, key: str) -> MatchResult | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{case_id}__{key}.json"
    if not path.exists():
        return None
    try:
        data = load_json(path)
        return MatchResult(
            list(data["matched_pred_indices"]),
            list(data["matched_truth_indices"]),
            list(data.get("reasons", [])),
        )
    except Exception:
        return None


def write_cache(
    cache_dir: Path | None,
    case_id: str,
    key: str,
    result: MatchResult,
    prompt: str,
    raw_response: str,
) -> None:
    if cache_dir is None:
        return
    save_json(
        cache_dir / f"{case_id}__{key}.json",
        {
            "version": PROMPT_VERSION,
            "matched_pred_indices": result.matched_pred_indices,
            "matched_truth_indices": result.matched_truth_indices,
            "reasons": result.reasons,
            "prompt": prompt,
            "raw_response": raw_response,
        },
    )


def score_with_judge(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = load_cases(args.dataset)
    files = result_files(args.results)
    if not files:
        raise RuntimeError(f"No case result JSON files found in {args.results}")

    prepared = []
    case_scores: dict[str, dict[str, Any]] = {}
    judge_results: dict[str, MatchResult] = {}

    for result_file in files:
        case_id = result_file.stem
        case_data = cases.get(case_id)
        if case_data is None:
            print(f"skip {case_id}: not present in dataset", file=sys.stderr)
            continue
        truths = read_truths(case_data)
        preds = read_predictions(result_file)
        repo = str(case_data.get("repo", ""))

        if not preds or not truths:
            result = MatchResult([], [], [])
            judge_results[case_id] = result
            tp = 0
            fp = len(preds)
            fn = len(truths)
            case_scores[case_id] = compute_score(tp, fp, fn)
            continue

        key = cache_key(args.model, case_id, truths, preds)
        cached = read_cache(args.cache_dir, case_id, key)
        if cached is not None:
            judge_results[case_id] = cached
            tp = len(set(cached.matched_pred_indices))
            fp = len(preds) - tp
            fn = len(truths) - len(set(cached.matched_truth_indices))
            case_scores[case_id] = compute_score(tp, fp, fn)
            continue

        prepared.append((case_id, repo, truths, preds, key))

    for start in range(0, len(prepared), args.batch_size):
        batch_items = prepared[start:start + args.batch_size]
        batch = [(case_id, repo, truths, preds) for case_id, repo, truths, preds, _ in batch_items]
        batch_no = start // args.batch_size + 1
        total_batches = (len(prepared) + args.batch_size - 1) // args.batch_size
        print(
            f"Judge batch {batch_no}/{total_batches}: "
            f"{', '.join(case_id for case_id, _, _, _, _ in batch_items)}",
            flush=True,
        )

        prompt = build_prompt(batch)
        raw = call_codex(prompt, args.model, args.timeout)
        payload = extract_json(raw)
        cases_out = payload.get("cases", {})
        if not isinstance(cases_out, dict):
            cases_out = {}

        for case_id, _repo, truths, preds, key in batch_items:
            entry = cases_out.get(case_id, {})
            matches = entry.get("matches", []) if isinstance(entry, dict) else []
            result = parse_matches(matches, len(preds), len(truths))
            judge_results[case_id] = result
            write_cache(args.cache_dir, case_id, key, result, prompt, raw)
            tp = len(set(result.matched_pred_indices))
            fp = len(preds) - tp
            fn = len(truths) - len(set(result.matched_truth_indices))
            case_scores[case_id] = compute_score(tp, fp, fn)

    metrics = aggregate(case_scores)
    return metrics, case_scores


def print_table(metrics: dict[str, Any], case_scores: dict[str, dict[str, Any]]) -> None:
    print("\nPer-case scores:")
    print(f"{'Case ID':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 74)
    for case_id in sorted(case_scores):
        score = case_scores[case_id]
        print(
            f"{case_id:<24} "
            f"{score['precision'] * 100:>9.2f}% "
            f"{score['recall'] * 100:>9.2f}% "
            f"{score['f1'] * 100:>9.2f}% "
            f"{score['tp']:>4} {score['fp']:>4} {score['fn']:>4}"
        )

    print("\nAggregate:")
    print(f"  Cases    : {metrics['cases']}")
    print(f"  Precision: {metrics['precision'] * 100:.2f}%")
    print(f"  Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"  F1       : {metrics['f1'] * 100:.2f}%")
    print(f"  TP/FP/FN : {metrics['total_tp']}/{metrics['total_fp']}/{metrics['total_fn']}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results = Path.cwd() / "results"

    parser = argparse.ArgumentParser(description="Score benchmark results with Codex as judge.")
    parser.add_argument("--dataset", type=Path, default=script_dir / "dataset.json")
    parser.add_argument("--results", type=Path, default=default_results)
    parser.add_argument("--scores-file", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.results = args.results.resolve()
    if args.scores_file is None:
        args.scores_file = args.results / "scores.json"
    else:
        args.scores_file = args.scores_file.resolve()
    args.cache_dir = None if args.no_cache else args.results / ".judge_cache"
    if args.batch_size < 1:
        args.batch_size = 1

    if not args.dataset.is_file():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    if not args.results.is_dir():
        print(f"results directory not found: {args.results}", file=sys.stderr)
        return 1

    print("Codex judge scoring")
    print(f"Dataset : {args.dataset}")
    print(f"Results : {args.results}")
    print(f"Model   : {args.model}")
    print(f"Cache   : {'disabled' if args.no_cache else args.cache_dir}")
    print(f"Batch   : {args.batch_size}")

    try:
        metrics, case_scores = score_with_judge(args)
    except Exception as exc:
        print(f"score failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "metrics": metrics,
        "case_scores": case_scores,
        "judge": {
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
        },
    }
    save_json(args.scores_file, payload)
    print_table(metrics, case_scores)
    print(f"\nScores written to {args.scores_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
