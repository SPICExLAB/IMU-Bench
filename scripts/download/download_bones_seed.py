from huggingface_hub import snapshot_download

# Download the full dataset
snapshot_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    local_dir="./bones-seed"
)
