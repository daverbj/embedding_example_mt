# PyTorch Sentence Embedding Client

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance C++ client for generating text embeddings using exported PyTorch sentence transformer models. Includes both single-threaded and multi-threaded implementations with real-time progress visualization.

## Features

- **Single-threaded Client**: Simple, reliable embedding generation for sequential processing
- **Multi-threaded Client**: Parallel processing with configurable thread count
- **Real-time Progress Visualization**: Color-coded progress bars showing embedding status
- **JSON Output**: Structured output of embeddings for easy integration with other systems
- **Memory Efficient**: Optimized tensor handling for reduced memory usage
- **Cross-platform**: Tested on macOS, should work on Linux and Windows with proper PyTorch setup

## Prerequisites

- C++17 compatible compiler (GCC, Clang, MSVC)
- PyTorch C++ libraries (LibTorch)
- Python with PyTorch and Transformers libraries (`pip install transformers torch`)

## Step 1: Export a PyTorch Model

If you don't have an exported model yet, you can use the included script:

```bash
python export_model.py
```

This will:
1. Download the Hugging Face model (default is 'sentence-transformers/all-MiniLM-L6-v2')
2. Convert it to TorchScript format
3. Save it as `embeddings_model.pt`
4. Save tokenizer configuration as `tokenizer_config.json`

## Step 2: Build the Clients

### Build the Single-threaded Client

```bash
./build_with_clang.sh
```

### Build the Multi-threaded Client

```bash
./build_mt_client.sh
```

## Step 3: Run the Clients

### Single-threaded Client

```bash
./build_mac/embedding_client
```

### Multi-threaded Client

```bash
./build_mt2/multithread_client --input sentences.txt --output embeddings_output.json
```

### Command Line Options (Multi-threaded Client)

- `--model PATH`: Path to the model file (default: embeddings_model.pt)
- `--input PATH`: Path to input sentences file (default: sentences.txt)
- `--output PATH`: Path to output embeddings file (default: embeddings_output.json)
- `--threads N`: Number of worker threads (default: number of CPU cores)
- `--help`: Show help message

## How It Works

The C++ application:
1. Loads the TorchScript model
2. Tokenizes input text (using a simplified tokenizer)
3. Passes the tokenized input to the model to generate embeddings
4. In multi-threaded mode, distributes the work across multiple threads with progress visualization
5. Saves the resulting embeddings in JSON format

## Demo

When running the multi-threaded client, you'll see a real-time progress display like this:

```
Thread 1 |#####################-------------------| 55.5% Complete
Thread 2 |####################--------------------| 50.0% Complete
Thread 3 |#######################-----------------| 65.0% Complete
Thread 4 |########################----------------| 60.0% Complete

Overall progress: 57.6%
```

## Implementation Details

### Model Export

The `export_model.py` script converts a Hugging Face sentence transformer model to TorchScript format:

```python
# Export the model using torch.jit.trace with empty inputs
sample_text = ""
encoded = self.tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

with torch.no_grad():
    traced_model = torch.jit.trace(
        self,
        (input_ids, attention_mask)
    )
```

### Tensor Handling in C++

The client uses `torch::from_blob()` for efficient tensor creation:

```cpp
torch::Tensor input_ids_tensor = torch::from_blob(padded_ids.data(), 
    {1, static_cast<long>(padded_ids.size())}, 
    torch::kInt64).clone().to(device);
```

### Multi-threading Architecture

The multi-threaded client uses:
- `EmbeddingWorker` class for processing batches of sentences
- `ProgressTracker` for monitoring and displaying progress
- `ProgressBar` for visualizing individual thread progress

## Notes

- The tokenization in the C++ code is simplified. For production use, you should implement a proper tokenizer that matches the one used during model training.
- You can easily adapt the client to work with different PyTorch-exported models by modifying the tokenization and input processing.

## Troubleshooting

- If you see errors about missing libraries, make sure PyTorch is properly installed
- If the model fails to load, check that the export step completed successfully
- For tokenization issues, you might need to implement a more sophisticated tokenizer

## Project Structure

- `embedding_client.cpp`: Single-threaded client implementation
- `multithread_client.cpp`: Multi-threaded client with progress visualization
- `export_model.py`: Script to export PyTorch models for C++ consumption
- `build_with_clang.sh`: Build script for single-threaded client
- `build_mt_client.sh`: Build script for multi-threaded client
- `sentences.txt`: Sample input sentences
- `embeddings_model.pt`: Exported PyTorch model
- `tokenizer_config.json`: Tokenizer configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the excellent C++ API
- [Hugging Face](https://huggingface.co/) for the sentence transformer models
- [Sentence Transformers](https://www.sbert.net/) for the original Python implementation
