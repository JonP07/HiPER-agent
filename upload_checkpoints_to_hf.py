#!/usr/bin/env python3
"""Upload model checkpoint files to a Hugging Face model repository.

Examples:
  python upload_checkpoints_to_hf.py \
    --repo-id your-username/your-model-repo \
    --local-path ./checkpoints/my_run \
    --path-in-repo checkpoints/my_run

  python upload_checkpoints_to_hf.py \
    --repo-id your-username/your-model-repo \
    --local-path ./checkpoints/best_model.pt \
    --path-in-repo checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local checkpoint file/folder to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target model repo in the form 'namespace/repo_name'.",
    )
    parser.add_argument(
        "--local-path",
        required=True,
        help="Path to a local file or directory to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="checkpoints",
        help="Destination path inside the Hugging Face repo.",
    )
    parser.add_argument(
        "--token",
        default="HF_TOKEN_PLACEHOLDER",
        help=(
            "Hugging Face token. Keep placeholder in code and pass real token at runtime, "
            "or set HF_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload model checkpoints",
        help="Commit message used for upload.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to upload to.",
    )
    return parser.parse_args()


def resolve_token(cli_token: str) -> str:
    env_token = os.getenv("HF_TOKEN")
    token = env_token or cli_token
    if not token or token == "HF_TOKEN_PLACEHOLDER":
        raise ValueError(
            "No valid Hugging Face token provided. Set HF_TOKEN or pass --token <your_token>."
        )
    return token


def ensure_repo(api: HfApi, repo_id: str, token: str, private: bool) -> None:
    # Safe when repo already exists.
    api.create_repo(repo_id=repo_id, repo_type="model", token=token, private=private, exist_ok=True)


def upload_path(
    api: HfApi,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
    token: str,
    commit_message: str,
    revision: str,
) -> None:
    if local_path.is_dir():
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=path_in_repo,
            token=token,
            commit_message=commit_message,
            revision=revision,
        )
    else:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=commit_message,
            revision=revision,
        )


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)

    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    api = HfApi()
    ensure_repo(api=api, repo_id=args.repo_id, token=token, private=args.private)

    upload_path(
        api=api,
        repo_id=args.repo_id,
        local_path=local_path,
        path_in_repo=args.path_in_repo,
        token=token,
        commit_message=args.commit_message,
        revision=args.revision,
    )

    print("Upload complete.")
    print(f"Repo: {args.repo_id}")
    print(f"Uploaded: {local_path}")
    print(f"Destination: {args.path_in_repo}")


if __name__ == "__main__":
    main()
