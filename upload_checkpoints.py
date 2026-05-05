from huggingface_hub import HfApi

api = HfApi()
repo_id = "Prajanya23/Coda"

files_to_upload = [
    (
        "/scratch/pg2963/bach-gpt/results/compound_10k_v2/checkpoints_compound/compound_best.pt",
        "checkpoints/compound_best.pt"
    ),
    (
        "/scratch/pg2963/bach-gpt/results/compound_10k_v2/checkpoints_contrastive_compound/clap_compound_best.pt",
        "checkpoints/clap_compound_best.pt"
    ),
    (
        "/scratch/pg2963/bach-gpt/results/compound_10k_v2/checkpoints_prefix/prefix_projector_best.pt",
        "checkpoints/prefix_projector_best.pt"
    ),
]

for local_path, repo_path in files_to_upload:
    print(f"Uploading {local_path} → {repo_path} ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done: {repo_path}")