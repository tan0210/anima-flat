import argparse
import getpass
import os
import re
import sys

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download
from safetensors import safe_open

from ..config import FORMAT
from ..lora import LORA_FORMAT

KINDS = {FORMAT: "model checkpoint", LORA_FORMAT: "LoRA adapter"}


def file_kind(path: str) -> str:
    """Identify an anima-flat file.

    Args:
        path: A safetensors file.

    Returns:
        `"model checkpoint"` or `"LoRA adapter"`.

    Raises:
        ValueError: If the file is missing or is neither an `anima-flat-1` checkpoint nor an
            `anima-flat-lora-1` adapter.
    """
    if not os.path.isfile(path):
        raise ValueError(f"{path}: no such file")
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            fmt = (f.metadata() or {}).get("format")
    except Exception as e:
        raise ValueError(f"{path}: not a readable safetensors file ({e})") from e
    if fmt not in KINDS:
        raise ValueError(f"{path}: format {fmt!r} is not one of {sorted(KINDS)}")
    return KINDS[fmt]


def resolve_token(arg: str | None) -> str:
    """Find a Hugging Face token.

    Looks at `--token`, then the `HF_TOKEN` environment variable and the cached `hf auth login`
    token, and finally asks on the terminal (input is hidden).

    Args:
        arg: Token from the command line, if any.

    Returns:
        The token.

    Raises:
        SystemExit: If none is found and there is no terminal to ask on.
    """
    token = arg or get_token()
    if token:
        return token
    if sys.stdin.isatty():
        token = getpass.getpass("Hugging Face token (needs write access): ").strip()
        if token:
            return token
    raise SystemExit(
        "no Hugging Face token: pass --token, set HF_TOKEN, or run `hf auth login`"
    )


def inherit_license(repo: str) -> tuple[dict[str, str], bytes, bytes]:
    """Take the license of another model repo, as its derivatives must carry it.

    Args:
        repo: Repo with a `LICENSE.md` and a model card whose front matter names the license.

    Returns:
        The license fields for the new model card (`license`, `license_name`, `license_link`),
        the license text, and the attribution notice the license requires, as a `NOTICE.md` body.
        The notice ends with a statement that the weights were modified, as derivatives must say.

    Raises:
        ValueError: If the repo has no license metadata or the license has no attribution notice.
    """
    with open(hf_hub_download(repo, "README.md"), encoding="utf-8") as f:
        text = f.read()
    front = yaml.safe_load(text.split("---")[1]) if text.startswith("---") else {}
    if not front.get("license"):
        raise ValueError(f"{repo} has no license in its model card")
    with open(hf_hub_download(repo, "LICENSE.md"), "rb") as f:
        lic = f.read()
    m = re.search(
        "\u201c(The .{0,80} Model is licensed by .*?)\u201d",
        lic.decode("utf-8"),
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"{repo}/LICENSE.md contains no attribution notice")
    meta = {"license": front["license"], "license_link": "LICENSE.md"}
    if front.get("license_name"):
        meta["license_name"] = front["license_name"]
    notice = (
        f"{m.group(1).strip()}\n\n"
        f"These weights are a modified version of {repo}, converted or fine-tuned with anima-flat.\n"
    )
    return meta, lic, notice.encode("utf-8")


def model_card(
    args: argparse.Namespace,
    files: list[tuple[str, str, int]],
    license_meta: dict[str, str] | None = None,
) -> str:
    """Write a minimal model card.

    Args:
        args: Options with `base_model` and `license`.
        files: `(name in repo, kind, size in bytes)` for each uploaded file.
        license_meta: Front-matter license fields from `inherit_license`; overrides `args.license`.

    Returns:
        Markdown with YAML front matter.
    """
    head = [
        "---",
        f"base_model: {args.base_model}",
        "tags:",
        "- anima",
        "- text-to-image",
    ]
    if license_meta:
        head += [f"{k}: {v}" for k, v in license_meta.items()]
    elif args.license:
        head.append(f"license: {args.license}")
    head.append("---")
    rows = "\n".join(f"| `{n}` | {k} | {s / 2**30:.2f} GiB |" for n, k, s in files)
    ckpt = next(
        (n for n, k, _ in files if k == "model checkpoint"), "anima-flat.safetensors"
    )
    lora = next((n for n, k, _ in files if k == "LoRA adapter"), None)
    use = (
        f'model = load("{ckpt}")'
        if lora is None
        else f'model = load("{ckpt}", lora="{lora}")'
    )
    lic = (
        "\nThese weights are modified derivatives of Anima and carry its license, which allows "
        "non-commercial use only. See `LICENSE.md` and `NOTICE.md`.\n"
        if license_meta
        else ""
    )
    body = f"""
Weights for [anima-flat](https://github.com/tan0210/anima-flat), a flattened inference and
fine-tuning backend for Anima ({args.base_model}).

| file | kind | size |
|---|---|---|
{rows}
{lic}
```python
from src.model import load

{use}
```
"""
    return "\n".join(head) + "\n" + body


def main(argv: list[str] | None = None) -> int:
    """Upload anima-flat checkpoints or LoRA adapters to a Hugging Face repo.

    Every file is checked to be an `anima-flat-1` checkpoint or an `anima-flat-lora-1` adapter
    before anything is sent, and all files go up in one commit. The repo is created if it does
    not exist (private unless `--public`). A model card is added only when the repo has none.

    Args:
        argv: Command-line arguments; defaults to `sys.argv[1:]`.

    Returns:
        0 on success, 1 if a file is invalid.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("files", nargs="+", help="Safetensors files to upload.")
    ap.add_argument(
        "--repo-id",
        required=True,
        help="`user/name`; a bare name is placed under your account.",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="Create the repo as public (default private).",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="Defaults to HF_TOKEN, the cached login, or a prompt.",
    )
    ap.add_argument(
        "--branch", default=None, help="Upload to this branch (created if missing)."
    )
    ap.add_argument(
        "--subdir", default="", help="Folder inside the repo to place the files in."
    )
    ap.add_argument("--message", default="Upload anima-flat weights")
    ap.add_argument("--no-card", action="store_true", help="Do not add a model card.")
    ap.add_argument("--base-model", default="circlestone-labs/Anima")
    ap.add_argument(
        "--license", default=None, help="License id for the model card, e.g. `other`."
    )
    ap.add_argument(
        "--license-from",
        default=None,
        metavar="REPO",
        help="Copy this repo's license: uploads its LICENSE.md and the required "
        "NOTICE.md, and puts its license fields in the model card.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Validate and print the plan only."
    )
    args = ap.parse_args(argv)

    files = []
    for path in args.files:
        try:
            kind = file_kind(path)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        name = "/".join(
            p for p in (args.subdir.strip("/"), os.path.basename(path)) if p
        )
        files.append((path, name, kind, os.path.getsize(path)))

    names = [n for _, n, _, _ in files]
    if len(set(names)) != len(names):
        print("error: two files would have the same name in the repo", file=sys.stderr)
        return 1

    license_meta, lic, notice = None, b"", b""
    if args.license_from:
        try:
            license_meta, lic, notice = inherit_license(args.license_from)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

    visibility = "public" if args.public else "private"
    print(
        f"{args.repo_id} ({visibility})"
        + (f", branch {args.branch}" if args.branch else "")
    )
    for path, name, kind, size in files:
        print(f"  {name}  {kind}, {size / 2**30:.2f} GiB")
    if license_meta:
        print(
            f"  LICENSE.md, NOTICE.md  from {args.license_from} ({license_meta['license']})"
        )
    if args.dry_run:
        return 0

    api = HfApi(token=resolve_token(args.token))
    user = api.whoami()["name"]
    repo_id = args.repo_id if "/" in args.repo_id else f"{user}/{args.repo_id}"
    api.create_repo(repo_id, private=not args.public, exist_ok=True)
    existing = api.repo_info(repo_id).private
    if existing != (not args.public):
        print(
            f"note: the repo already exists as {'private' if existing else 'public'}; visibility unchanged"
        )
    if args.branch:
        api.create_branch(repo_id, branch=args.branch, exist_ok=True)

    ops = [
        CommitOperationAdd(path_in_repo=n, path_or_fileobj=p) for p, n, _, _ in files
    ]
    if license_meta:
        ops.append(CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj=lic))
        ops.append(CommitOperationAdd(path_in_repo="NOTICE.md", path_or_fileobj=notice))
    if not args.no_card and not api.file_exists(
        repo_id, "README.md", revision=args.branch
    ):
        card = model_card(args, [(n, k, s) for _, n, k, s in files], license_meta)
        ops.append(
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())
        )
    info = api.create_commit(
        repo_id, ops, commit_message=args.message, revision=args.branch
    )
    print(
        f"https://huggingface.co/{repo_id}"
        + (f"/tree/{args.branch}" if args.branch else "")
    )
    print(info.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
