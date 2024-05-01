#!/bin/bash

HF_USER="BramVanroy"
OLLAMA_USER="bramvanroy"

set -e

usage() {
    echo "Usage: $0 -m MODEL_NAME -o OUTPUT_DIR [-f FORMAT]"
    echo "  -m, --model-name      Model identifier on Huggingface (required)"
    echo "  -o, --output-dir      Output directory where model will be saved (required)"
    echo "  -t, --template-file   Template file for the Modelfile, including for instance stop tokens or system message. FROM will be inserted automatically at the top so do not include that."
    echo "  -f, --format          Model format for initial model, options are: f16 or f32 (default: f16)"
    exit 1
}

# Default format
FORMAT="f16"

# Parse command-line options
while getopts "m:o:t:f:h" opt; do
    case "$opt" in
    m) MODEL_NAME="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    t) TEMPLATE_FILE="$OPTARG" ;;
    f) FORMAT="$OPTARG" ;;
    h) usage ;;
    ?) usage ;;
    esac
done

# Check for required parameters
if [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: MODEL_NAME and OUTPUT_DIR are required."
    usage
fi

# Validate format
if [ "$FORMAT" != "f16" ] && [ "$FORMAT" != "f32" ]; then
    echo "Error: Invalid format specified. Choose between 'f16' or 'f32'."
    usage
fi

mkdir -p "$OUTPUT_DIR"

# Download model if directory is empty
if [ -z "$(ls -A $OUTPUT_DIR)" ]; then
    # Download model using Python
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_NAME', local_dir='$OUTPUT_DIR', local_dir_use_symlinks=False)"
    echo "Downloaded $MODEL_NAME to $OUTPUT_DIR"
else
    echo "$OUTPUT_DIR is not empty, so skipping download"
fi

# Convert model to GGUF in given outtype (defaults to f16)
python convert-hf-to-gguf.py "$OUTPUT_DIR" --outtype "$FORMAT"

# Set SHORT_MODEL_NAME
SHORT_MODEL_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | awk -F'/' '{print $NF}')

# Move and rename the base version to the correct location
BASE_VERSION_GGUF="$OUTPUT_DIR/$FORMAT/$SHORT_MODEL_NAME-$FORMAT.gguf"
mkdir -p "$OUTPUT_DIR/$FORMAT/"
mv "$OUTPUT_DIR/ggml-model-$FORMAT.gguf" "$BASE_VERSION_GGUF"
echo "Moved $OUTPUT_DIR/ggml-model-$FORMAT.gguf to $BASE_VERSION_GGUF"

# Do the quantization for the different quant types and upload to Huggingface and ollama
for quant_type in Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0 f16 f32; do
    # Skip quantization if the quant type is f32 and the format is f16
    if [[ "$quant_type" == "f32" && "$FORMAT" == "f16" ]]; then
        continue
    fi

    mkdir -p "$OUTPUT_DIR/$quant_type/"
    output_file_gguf="$OUTPUT_DIR/$quant_type/$SHORT_MODEL_NAME-$quant_type.gguf"

    # Quantize if the base version is not the same as the output file (because the base version is already quantized)
    if [ "$BASE_VERSION_GGUF" != "$output_file_gguf" ]; then
        echo "Quantizing: build/bin/quantize $BASE_VERSION_GGUF $output_file_gguf $quant_type"
        build/bin/quantize "$BASE_VERSION_GGUF" "$output_file_gguf" "$quant_type"
    else
        echo "Skipping quantization for input format $FORMAT and current quanttype $quant_type"
    fi

    # Create Modelfile based on given template file, if it is given
    modelfile="$OUTPUT_DIR/$quant_type/Modelfile"
    content=""
    if [ -f "$TEMPLATE_FILE" ]; then
        content=$(cat "$TEMPLATE_FILE")
    fi
    content="FROM ./$SHORT_MODEL_NAME-$quant_type.ggufn$content"
    echo -e "$content" >"$modelfile"

    # Create ollama model and upload
    echo "Ollama create: ollama create $OLLAMA_USER/$SHORT_MODEL_NAME:$quant_type -f $modelfile"
    ollama create "$OLLAMA_USER/$SHORT_MODEL_NAME:$quant_type" -f "$modelfile"

    echo "Ollama push: ollama push $OLLAMA_USER/$SHORT_MODEL_NAME:$quant_type"
    ollama push "$OLLAMA_USER/$SHORT_MODEL_NAME:$quant_type"

    # Upload to Hugging Face repository, in its own quant directory
    echo "HF upload: huggingface-cli upload $HF_USER/${SHORT_MODEL_NAME}-GGUF $OUTPUT_DIR/$quant_type $quant_type/."
    huggingface-cli upload "$HF_USER/${SHORT_MODEL_NAME}-GGUF" "$OUTPUT_DIR/$quant_type" "$quant_type/."
done
