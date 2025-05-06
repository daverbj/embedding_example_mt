#!/bin/bash
# build_with_clang.sh - Script to build the embedding client on macOS with clang++

echo "Building embedding_client.cpp with clang++ on macOS..."

# Try to find the PyTorch installation
if command -v python3 &> /dev/null; then
    # Use Python to find torch paths
    TORCH_INCLUDE=$(python3 -c "import torch; print(torch.__path__[0] + '/include')")
    TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')")
    
    if [ -d "$TORCH_INCLUDE" ] && [ -d "$TORCH_LIB" ]; then
        echo "Found PyTorch at: $TORCH_INCLUDE"
    else
        # Try the Homebrew path as fallback
        echo "PyTorch not found in Python path, trying Homebrew paths..."
        TORCH_INCLUDE="/opt/homebrew/Cellar/pytorch/2.5.1_4/include"
        TORCH_LIB="/opt/homebrew/Cellar/pytorch/2.5.1_4/lib"
    fi
else
    echo "Python3 not found, using default Homebrew paths..."
    TORCH_INCLUDE="/opt/homebrew/Cellar/pytorch/2.5.1_4/include"
    TORCH_LIB="/opt/homebrew/Cellar/pytorch/2.5.1_4/lib"
fi

# Create build directory
mkdir -p build_mac
cd build_mac

# Build with clang++
clang++ -std=c++17 -O3 -DNDEBUG \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE/torch/csrc/api/include" \
    -L"$TORCH_LIB" \
    -Wl,-rpath,"$TORCH_LIB" \
    -o embedding_client ../embedding_client.cpp \
    -ltorch -lc10 -ltorch_cpu

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "The executable is at: build_mac/embedding_client"
    echo ""
    echo "To run the client:"
    echo "./build_mac/embedding_client"
    echo ""
    echo "You can also provide custom paths to the model and tokenizer config:"
    echo "./build_mac/embedding_client embeddings_model.pt tokenizer_config.json"
else
    echo "Build failed!"
fi

# Return to the original directory
cd ..
