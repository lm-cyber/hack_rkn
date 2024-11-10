from huggingface_hub import hf_hub_download

# Download the ONNX model
hf_hub_download(repo_id="alan3333/hack_rkn_onnx", filename="vit_v4.onnx", local_dir="./")
print("Model downloaded successfully.")
