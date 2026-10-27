"""One-command deploy of the Streamlit app to a Hugging Face Space.

Prerequisites (you do these once):
  1. Create a free account at https://huggingface.co/join
  2. Create a WRITE token at https://huggingface.co/settings/tokens
  3. Log in from this terminal:   huggingface-cli login
     (paste the WRITE token when asked)

Then run:   python deploy_to_hf.py
Optionally choose a name:   python deploy_to_hf.py --name my-fruit-app
"""
from __future__ import annotations
import argparse
import sys

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fresh-vs-rotten", help="Space name (under your account)")
    ap.add_argument("--private", action="store_true", help="Make the Space private")
    args = ap.parse_args()

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        print("ERROR: not logged in. Run `huggingface-cli login` with a WRITE token first.")
        sys.exit(1)

    user = who["name"]
    repo_id = f"{user}/{args.name}"
    print(f"Deploying as: {user}")
    print(f"Space: {repo_id}")

    # Create the Space as a Docker Space (runs the Streamlit app via the
    # Dockerfile in app/). HF's API no longer accepts "streamlit" as an SDK.
    # Safe to re-run.
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)

    # Upload the app/ folder contents to the Space root.
    print("Uploading app/ (this includes the trained model, ~6 MB) ...")
    api.upload_folder(
        folder_path="app",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy fresh-vs-rotten classifier",
        ignore_patterns=["__pycache__*", "*.pyc", "PLACEHOLDER.md"],
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\nDone! Your Space is building now and will be live in a few minutes at:")
    print("  " + url)
    print("Watch the build logs on that page; the first build takes a few minutes.")


if __name__ == "__main__":
    main()
