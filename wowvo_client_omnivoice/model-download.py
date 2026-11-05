from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepdml/faster-whisper-large-v3-turbo-ct2",
    local_dir=r"I:\voiceover\model\Faster-Whisper",
    local_dir_use_symlinks=False  # реальные файлы, а не симлинки
)