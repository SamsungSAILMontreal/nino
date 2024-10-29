# This scripts downloads the dataset manually from the Hugging Face hub
# If this script is not run before the training script, the code in dataset.py will download the dataset automatically
# Usage: bash download.sh <cache_dir>, where <cache_dir> is the directory where the dataset will be downloaded
# Note: the <cache_dir> directory should have about 200GB of free space
huggingface-cli download SamsungSAILMontreal/nino_metatrain --repo-type dataset --revision mmap --cache-dir "$1"