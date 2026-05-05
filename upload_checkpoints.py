from huggingface_hub import HfApi

api = HfApi()
repo_id = "Prajanya23/Coda"

# Upload the two main checkpoints + tokenizer vocab
files_to_upload = [
    "/scratch/pg2963/bach-gpt/results/compound_10k_v2/checkpoint_best.pt",
    "/scratch/pg2963/bach-gpt/results/smoke_test_v1/checkpoint_best.pt",
    "/scratch/pg2963/bach-gpt/data/vocab.json",  # your tokenizer vocab
]

for path in files_to_upload:
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path.split("/")[-1],  # flat structure in the repo
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {path}")
