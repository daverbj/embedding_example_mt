#!/bin/bash


echo "Building multithread_client.cpp with clang++ on macOS..."

if command -v python3 &> /dev/null; then
  
    TORCH_INCLUDE=$(python3 -c "import torch; print(torch.__path__[0] + '/include')")
    TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')")
    
    if [ -d "$TORCH_INCLUDE" ] && [ -d "$TORCH_LIB" ]; then
        echo "Found PyTorch at: $TORCH_INCLUDE"
    else
        
        echo "PyTorch not found in Python path, trying Homebrew paths..."
        TORCH_INCLUDE="/opt/homebrew/Cellar/pytorch/2.5.1_4/include"
        TORCH_LIB="/opt/homebrew/Cellar/pytorch/2.5.1_4/lib"
    fi
else
    echo "Python3 not found, using default Homebrew paths..."
    TORCH_INCLUDE="/opt/homebrew/Cellar/pytorch/2.5.1_4/include"
    TORCH_LIB="/opt/homebrew/Cellar/pytorch/2.5.1_4/lib"
fi


mkdir -p build_mt2
cd build_mt2


clang++ -v -std=c++17 -O3 \
    -I"$TORCH_INCLUDE" \
    -I"$TORCH_INCLUDE/torch/csrc/api/include" \
    -L"$TORCH_LIB" \
    -Wl,-rpath,"$TORCH_LIB" \
    -o multithread_client ../multithread_client.cpp \
    -ltorch -lc10 -ltorch_cpu


if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "The executable is at: build_mt2/multithread_client"
    echo ""
    echo "To run the client:"
    echo "./build_mt2/multithread_client"
else
    echo "Build failed!"
    
  
    echo ""
    echo "Debugging information:"
    echo "---------------------"
    echo "PyTorch include path: $TORCH_INCLUDE"
    echo "PyTorch library path: $TORCH_LIB"
    
  
    echo ""
    echo "Checking for essential files:"
    if [ -f "$TORCH_INCLUDE/torch/torch.h" ]; then
        echo "✓ torch.h found"
    else
        echo "✗ torch.h not found"
    fi
    
    if [ -f "$TORCH_INCLUDE/torch/csrc/api/include/torch/script.h" ]; then
        echo "✓ script.h found"
    else
        echo "✗ script.h not found"
    fi
    
    if [ -f "$TORCH_LIB/libtorch.dylib" ]; then
        echo "✓ libtorch.dylib found"
    else
        echo "✗ libtorch.dylib not found. Checking alternatives..."
        ls -la "$TORCH_LIB"/libtorch* 2>/dev/null || echo "No libtorch* files found."
    fi
fi

cd ..
