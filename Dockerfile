# syntax=docker/dockerfile:1.7
#
# Build with docker/ as the only context:
# docker build -f docker/Dockerfile-inline docker --build-arg CASES_PER_REPO=2 -t benchmark-dataset:2-per-repo-inline

FROM python:3.14.4-alpine AS builder

ARG HF_DATASET=Qodo/PR-Review-Bench
ARG HF_TOKEN=
ARG GITHUB_TOKEN=
ARG CASES_PER_REPO=1
ARG FETCH_DIFFS=1
ARG CHECKOUT_REPOS=1

WORKDIR /dataset

RUN apk add --no-cache ca-certificates git
RUN pip install --no-cache-dir -U "huggingface_hub"

RUN set -eu; \
    if [ -n "$HF_TOKEN" ]; then \
      hf download "$HF_DATASET" --repo-type dataset --local-dir /dataset --token "$HF_TOKEN"; \
    else \
      hf download "$HF_DATASET" --repo-type dataset --local-dir /dataset; \
    fi; \
    rm -rf /dataset/.cache

RUN python - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path


DATASET_FILE = Path("/dataset/git_code_review_bench_100_w_open_prs.jsonl")
RULES_FILE = Path("/dataset/rules_for_repo.jsonl")
DIFFS_DIR = Path("/dataset/diffs")
MANIFEST_FILE = Path("/dataset/subset_manifest.json")


def read_jsonl(path):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(path, entries):
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def pr_number(entry):
    return int(entry["pr_url_to_review"].rstrip("/").split("/")[-1])


def case_id(entry):
    return f"{entry['repo']}_{pr_number(entry)}"


def selected_case_manifest(entry):
    pr = pr_number(entry)
    return {
        "case_id": f"{entry['repo']}_{pr}",
        "repo": entry["repo"],
        "pr_number": pr,
        "pr_url": entry["pr_url_to_review"],
    }


def cases_per_repo():
    try:
        value = int(os.environ["CASES_PER_REPO"])
    except (KeyError, ValueError) as exc:
        raise SystemExit("CASES_PER_REPO must be a positive integer") from exc

    if value < 1:
        raise SystemExit("CASES_PER_REPO must be a positive integer")

    return value


def select_entries(entries, limit):
    entries_by_repo = defaultdict(list)
    repo_order = []

    for entry in entries:
        repo = entry["repo"]
        if repo not in entries_by_repo:
            repo_order.append(repo)
        entries_by_repo[repo].append(entry)

    selected = []
    for repo in repo_order:
        selected.extend(sorted(entries_by_repo[repo], key=pr_number)[:limit])

    return selected, repo_order


def filter_rules(selected_repos):
    if not RULES_FILE.exists():
        return

    rules = [
        entry
        for entry in read_jsonl(RULES_FILE)
        if entry.get("repo") in selected_repos
    ]
    write_jsonl(RULES_FILE, rules)


def prune_diffs(selected_ids):
    if not DIFFS_DIR.exists():
        return

    for diff_file in DIFFS_DIR.glob("*.diff"):
        if diff_file.stem not in selected_ids:
            diff_file.unlink()


def write_manifest(selected_cases, selected_repos, limit):
    manifest = {
        "cases_per_repo": limit,
        "total_cases": len(selected_cases),
        "repos": sorted(selected_repos),
        "cases": selected_cases,
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    limit = cases_per_repo()
    entries = read_jsonl(DATASET_FILE)
    selected, repo_order = select_entries(entries, limit)
    selected_cases = [selected_case_manifest(entry) for entry in selected]
    selected_repos = {entry["repo"] for entry in selected}
    selected_ids = {case_id(entry) for entry in selected}

    write_jsonl(DATASET_FILE, selected)
    filter_rules(selected_repos)
    prune_diffs(selected_ids)
    write_manifest(selected_cases, selected_repos, limit)

    print(f"Selected {len(selected_cases)} cases from {len(selected_repos)} repos")
    for repo in repo_order:
        count = sum(1 for case in selected_cases if case["repo"] == repo)
        print(f"  {repo}: {count}")


if __name__ == "__main__":
    main()
PY

RUN if [ "$FETCH_DIFFS" != "1" ]; then exit 0; fi; python - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


DATASET_FILE = Path("/dataset/git_code_review_bench_100_w_open_prs.jsonl")
DIFFS_DIR = Path("/dataset/diffs")


def read_entries():
    with DATASET_FILE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pr_number(entry):
    return int(entry["pr_url_to_review"].rstrip("/").split("/")[-1])


def diff_url(entry):
    path = entry["pr_url_to_review"].removeprefix("https://github.com/")
    return f"https://patch-diff.githubusercontent.com/raw/{path}.diff"


def request_headers():
    headers = {"User-Agent": "ai-code-review-benchmark-dataset-image"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_diff(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    DIFFS_DIR.mkdir(parents=True, exist_ok=True)

    entries = read_entries()
    headers = request_headers()
    errors = []

    for index, entry in enumerate(entries, start=1):
        repo = entry["repo"]
        pr = pr_number(entry)
        case_id = f"{repo}_{pr}"
        out_file = DIFFS_DIR / f"{case_id}.diff"

        if out_file.exists() and out_file.stat().st_size > 0:
            print(f"[{index:3d}/{len(entries)}] skip  {case_id}")
            continue

        try:
            content = fetch_diff(diff_url(entry), headers)
            out_file.write_bytes(content)
            print(f"[{index:3d}/{len(entries)}] ok    {case_id} ({len(content):,} bytes)")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[{index:3d}/{len(entries)}] ERROR {case_id}: {exc}")
            errors.append(case_id)

        time.sleep(0.2)

    if errors:
        raise SystemExit(f"Failed to fetch diffs: {', '.join(errors)}")


if __name__ == "__main__":
    main()
PY

RUN if [ "$CHECKOUT_REPOS" != "1" ]; then mkdir -p /dataset/repos; exit 0; fi; python - <<'PY'
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path


DATASET_FILE = Path("/dataset/git_code_review_bench_100_w_open_prs.jsonl")
REPOS_DIR = Path("/dataset/repos")
GITHUB_ORG = "agentic-review-benchmarks"


def load_repo_prs():
    repo_prs = defaultdict(list)
    with DATASET_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            pr_number = int(entry["pr_url_to_review"].rstrip("/").split("/")[-1])
            repo_prs[entry["repo"]].append(pr_number)
    return repo_prs


def run_git(repo_dir, args, check=True, auth=False):
    token = os.environ.get("GITHUB_TOKEN", "")
    command = ["git"]
    if auth and token:
        command.extend(["-c", f"http.https://github.com/.extraheader=Authorization: token {token}"])
    command.extend(args)

    result = subprocess.run(command, cwd=repo_dir, text=True, capture_output=True)
    if check and result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {details}")
    return result


def ensure_repo(repo):
    repo_dir = REPOS_DIR / repo
    remote_url = f"https://github.com/{GITHUB_ORG}/{repo}.git"

    repo_dir.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").exists():
        run_git(repo_dir, ["init"])
        run_git(repo_dir, ["remote", "add", "origin", remote_url])
        run_git(repo_dir, ["config", "remote.origin.promisor", "true"])
        run_git(repo_dir, ["config", "remote.origin.partialclonefilter", "blob:none"])
    else:
        run_git(repo_dir, ["remote", "set-url", "origin", remote_url])

    return repo_dir


def ensure_worktree(repo_dir, pr_number):
    worktree_dir = repo_dir / f"pr-{pr_number}"
    ref = f"refs/pr/{pr_number}"

    if worktree_dir.exists():
        result = run_git(worktree_dir, ["rev-parse", "--verify", "HEAD"], check=False)
        if result.returncode == 0:
            print(f"  PR #{pr_number}: already exists")
            return
        run_git(repo_dir, ["worktree", "remove", "--force", str(worktree_dir)])

    run_git(
        repo_dir,
        [
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            f"refs/pull/{pr_number}/head:{ref}",
        ],
        auth=True,
    )
    run_git(repo_dir, ["worktree", "add", f"pr-{pr_number}", ref])
    commit = run_git(worktree_dir, ["rev-parse", "HEAD"]).stdout.strip()
    print(f"  PR #{pr_number}: {commit[:12]}")


def main():
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    repo_prs = load_repo_prs()

    for repo in sorted(repo_prs):
        pr_numbers = sorted(set(repo_prs[repo]))
        repo_dir = ensure_repo(repo)
        print(f"{repo}: checking out {len(pr_numbers)} PR worktree(s)")
        for pr_number in pr_numbers:
            ensure_worktree(repo_dir, pr_number)


if __name__ == "__main__":
    main()
PY

RUN python - <<'PY'
import json
from pathlib import Path


DATASET_ROOT = Path("/dataset")
DATASET_FILE = DATASET_ROOT / "git_code_review_bench_100_w_open_prs.jsonl"
RULES_FILE = DATASET_ROOT / "rules_for_repo.jsonl"
OUTPUT_FILE = DATASET_ROOT / "dataset.json"


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rules():
    if not RULES_FILE.exists():
        return {}

    rules_by_repo = {}
    for entry in read_jsonl(RULES_FILE):
        rules_by_repo[entry["repo"]] = entry.get("extracted_rules", [])
    return rules_by_repo


def pr_number(entry):
    return int(entry["pr_url_to_review"].rstrip("/").split("/")[-1])


def index_entry(entry, rules_by_repo):
    repo = entry["repo"]
    pr = pr_number(entry)
    case_id = f"{repo}_{pr}"
    diff_file = Path("diffs") / f"{case_id}.diff"
    repo_dir = Path("repos") / repo / f"pr-{pr}"
    issues = entry.get("issues", [])

    return case_id, {
        "case_id": case_id,
        "repo": repo,
        "pr_number": pr,
        "pr_url": entry["pr_url_to_review"],
        "diff_file": diff_file.as_posix(),
        "repo_dir": repo_dir.as_posix(),
        "rules": rules_by_repo.get(repo, []),
        "issues": issues,
        "num_of_issues": entry.get("num_of_issues", len(issues)),
        "ready": {
            "diff": (DATASET_ROOT / diff_file).is_file(),
            "worktree": (DATASET_ROOT / repo_dir).is_dir(),
        },
    }


def main():
    rules_by_repo = load_rules()
    dataset = {}

    for entry in read_jsonl(DATASET_FILE):
        case_id, item = index_entry(entry, rules_by_repo)
        dataset[case_id] = item

    OUTPUT_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_FILE} with {len(dataset)} cases")


if __name__ == "__main__":
    main()
PY

FROM alpine:3.22

RUN apk add --no-cache ca-certificates git

WORKDIR /dataset
COPY --from=builder /dataset/ /dataset/

LABEL org.opencontainers.image.title="AI Code Review Benchmark Dataset"
LABEL org.opencontainers.image.description="Subset of Qodo PR-Review-Bench with the first N cases per repository"

CMD ["sh"]
