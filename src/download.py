import os
from huggingface_hub import snapshot_download, whoami

token = os.environ["HF_TOKEN"]
print(whoami(token=token))

snapshot_download(
    repo_id="Metacreation/GigaMIDI",
    repo_type="dataset",
    local_dir="./GigaMIDI",
    token=token
)
