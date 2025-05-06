#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <random>
#include <memory>
#include <functional>

// ANSI color codes for the progress bar
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

// Define a color list for different threads
const std::vector<std::string> COLORS = {GREEN, YELLOW, BLUE, MAGENTA, CYAN};

// Forward declarations
class ProgressBar;
class BatchLoader;
class EmbeddingWorker;
class ProgressTracker;

/**
 * Progress bar class for displaying task progress
 */
class ProgressBar {
private:
    size_t total;
    std::string prefix;
    std::string suffix;
    int decimals;
    int length;
    char fill;
    std::atomic<size_t> value;
    std::string color;

public:
    ProgressBar(size_t total, const std::string& prefix = "", 
               const std::string& suffix = "", int decimals = 1,
               int length = 50, char fill = '#', const std::string& color = WHITE)
        : total(total), prefix(prefix), suffix(suffix), 
          decimals(decimals), length(length), fill(fill), 
          value(0), color(color) {}

    void update(size_t new_value) {
        value.store(new_value);
    }

    std::string print() const {
        // Calculate the percentage
        float percent = static_cast<float>(value) / static_cast<float>(total) * 100.0f;
        
        // Calculate the filled portion of the bar
        int filled_length = static_cast<int>(length * value / total);
        
        // Create the bar
        std::string bar;
        for (int i = 0; i < length; ++i) {
            if (i < filled_length) {
                bar += fill;
            } else {
                bar += '-';
            }
        }
        
        // Format the percentage with specified decimals
        std::ostringstream percent_stream;
        percent_stream << std::fixed << std::setprecision(decimals) << percent;
        std::string percent_str = percent_stream.str();
        
        // Construct and return the progress bar string
        return color + "\r" + prefix + " |" + bar + "| " + percent_str + "% " + suffix + RESET;
    }
    
    bool is_complete() const {
        return value >= total;
    }
    
    size_t get_value() const {
        return value.load();
    }
    
    size_t get_total() const {
        return total;
    }
};

/**
 * BatchLoader class for loading sentences in batches
 */
class BatchLoader {
private:
    std::ifstream file;
    std::string file_path;
    size_t total_count;
    size_t processed_count;
    
public:
    BatchLoader(const std::string& file_path) : file_path(file_path), total_count(0), processed_count(0) {
        // First, count the total number of lines
        std::ifstream counting_file(file_path);
        if (counting_file.is_open()) {
            std::string line;
            while (std::getline(counting_file, line)) {
                if (!line.empty()) {
                    total_count++;
                }
            }
        } else {
            // If file doesn't exist, we'll generate 50 sample sentences
            total_count = 50;
        }
    }
    
    void open() {
        file.open(file_path);
        processed_count = 0;
    }
    
    bool is_open() const {
        return file.is_open();
    }
    
    void close() {
        if (file.is_open()) {
            file.close();
        }
    }
    
    size_t get_total_count() const {
        return total_count;
    }
    
    size_t get_processed_count() const {
        return processed_count;
    }
    
    // Load next batch of sentences
    std::vector<std::string> next_batch(size_t batch_size) {
        std::vector<std::string> batch;
        
        // If file couldn't be opened, generate sample sentences
        if (!file.is_open()) {
            size_t start = processed_count;
            size_t end = std::min(start + batch_size, total_count);
            for (size_t i = start; i < end; ++i) {
                batch.push_back("This is a sample sentence " + std::to_string(i+1) + 
                              " for embedding. It will be processed by a worker thread.");
            }
            processed_count = end;
            return batch;
        }
        
        // Load actual sentences from file
        std::string line;
        size_t count = 0;
        
        while (count < batch_size && std::getline(file, line)) {
            if (!line.empty()) {
                batch.push_back(line);
                count++;
                processed_count++;
            }
        }
        
        return batch;
    }
    
    // Check if there are more sentences to load
    bool has_more() const {
        if (!file.is_open()) {
            return processed_count < total_count;
        }
        return !file.eof();
    }
    
    // Reset to beginning of file
    void reset() {
        close();
        open();
        processed_count = 0;
    }
};

/**
 * Worker class that handles embedding sentences in batches
 */
class EmbeddingWorker {
private:
    int task_id;
    std::shared_ptr<BatchLoader> batch_loader;
    size_t batch_size;
    std::shared_ptr<torch::jit::script::Module> model;
    torch::Device device;
    std::unique_ptr<ProgressBar> progress_bar;
    std::atomic<bool> is_done;
    std::vector<torch::Tensor> results;
    std::vector<std::string> processed_sentences;
    std::mutex loader_mutex;
    size_t total_processed;
    
public:
    EmbeddingWorker(int task_id, 
                   std::shared_ptr<BatchLoader> batch_loader,
                   size_t batch_size,
                   size_t total_sentences,
                   std::shared_ptr<torch::jit::script::Module> model,
                   torch::Device device)
        : task_id(task_id), 
          batch_loader(batch_loader),
          batch_size(batch_size),
          model(model), 
          device(device), 
          is_done(false),
          total_processed(0) {
              
        // Create a progress bar with the corresponding color
        std::string color = COLORS[task_id % COLORS.size()];
        progress_bar = std::make_unique<ProgressBar>(
            total_sentences,
            "Thread " + std::to_string(task_id),
            "Complete",
            1, 30, '#', color
        );
        
        // Preallocate results vector
        results.reserve(total_sentences);
        processed_sentences.reserve(total_sentences);
    }
    
    // Tokenize a sentence (very basic implementation)
    std::pair<torch::Tensor, torch::Tensor> tokenize(const std::string& text) {
        // In a real implementation, this would use the tokenizer configuration
        // Here we just create dummy token ids for demonstration
        
        // Create sequence of 1s as token ids (this is just a placeholder)
        // In a real implementation, you would convert the text to actual token ids
        int seq_len = 10 + (text.length() % 10); // Simple way to get variable length
        
        // Create input ids tensor
        auto input_ids = torch::ones({1, seq_len}, torch::kInt64).to(device);
        
        // Create attention mask (1s for tokens, 0s for padding)
        auto attention_mask = torch::ones({1, seq_len}, torch::kInt64).to(device);
        
        return {input_ids, attention_mask};
    }
    
    // Main worker function to process sentences in batches
    void process() {
        bool has_more_work = true;
        
        while (has_more_work) {
            // Get the next batch of sentences
            std::vector<std::string> batch;
            {
                // Use mutex to ensure thread-safe access to batch loader
                std::lock_guard<std::mutex> lock(loader_mutex);
                if (batch_loader->has_more()) {
                    batch = batch_loader->next_batch(batch_size);
                } else {
                    has_more_work = false;
                }
            }
            
            // Process this batch
            for (const auto& sentence : batch) {
                // Get the token ids and attention mask
                auto [input_ids, attention_mask] = tokenize(sentence);
                
                // Create inputs for model
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_ids);
                inputs.push_back(attention_mask);
                
                // Run inference without gradients
                torch::NoGradGuard no_grad;
                torch::Tensor embedding = model->forward(inputs).toTensor();
                
                // Store the result
                results.push_back(embedding);
                processed_sentences.push_back(sentence);
                
                // Update progress
                total_processed++;
                progress_bar->update(total_processed);
                
                // Add some random delay to simulate variable processing time
                std::this_thread::sleep_for(std::chrono::milliseconds(50 + (std::rand() % 200)));
            }
            
            // If batch is empty and no more work, we're done
            if (batch.empty() && !has_more_work) {
                is_done = true;
                break;
            }
        }
        
        // Mark as done
        is_done = true;
    }
    
    // Getters
    bool done() const { return is_done; }
    const ProgressBar& get_progress_bar() const { return *progress_bar; }
    const std::vector<torch::Tensor>& get_results() const { return results; }
    const std::vector<std::string>& get_processed_sentences() const { return processed_sentences; }
    int get_id() const { return task_id; }
};

/**
 * ProgressTracker manages the display of all progress bars
 */
class ProgressTracker {
private:
    std::vector<std::shared_ptr<EmbeddingWorker>> workers;
    std::atomic<bool> stop_flag;
    std::thread display_thread;
    
    void display_loop() {
        // Clear terminal
        std::cout << "\033[2J\033[H";
        
        while (!stop_flag) {
            // Position cursor at start
            std::cout << "\033[H";
            
            // Calculate overall progress
            size_t total_done = 0;
            size_t total_work = 0;
            
            // Display each worker's progress bar
            for (const auto& worker : workers) {
                const auto& progress_bar = worker->get_progress_bar();
                std::cout << progress_bar.print() << std::endl;
                
                // Update overall statistics
                total_done += progress_bar.get_value();
                total_work += progress_bar.get_total();
            }
            
            // Display overall progress
            float overall_percent = (total_work > 0) ? 
                (static_cast<float>(total_done) / total_work * 100.0f) : 0.0f;
            
            std::cout << "\nOverall progress: " 
                      << std::fixed << std::setprecision(1) 
                      << overall_percent << "%" << std::endl;
            
            // Check if all workers are done
            bool all_done = true;
            for (const auto& worker : workers) {
                if (!worker->done()) {
                    all_done = false;
                    break;
                }
            }
            
            if (all_done) {
                stop_flag = true;
            }
            
            // Sleep for a short time
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Final update
        std::cout << "\033[H";
        for (const auto& worker : workers) {
            const auto& progress_bar = worker->get_progress_bar();
            std::cout << progress_bar.print() << std::endl;
        }
        
        std::cout << "\nAll embedding tasks completed!" << std::endl;
    }
    
public:
    ProgressTracker() : stop_flag(false) {}
    
    ~ProgressTracker() {
        if (display_thread.joinable()) {
            stop_flag = true;
            display_thread.join();
        }
    }
    
    void add_worker(std::shared_ptr<EmbeddingWorker> worker) {
        workers.push_back(worker);
    }
    
    void start() {
        display_thread = std::thread(&ProgressTracker::display_loop, this);
    }
    
    void stop() {
        stop_flag = true;
        if (display_thread.joinable()) {
            display_thread.join();
        }
    }
    
    const std::vector<std::shared_ptr<EmbeddingWorker>>& get_workers() const {
        return workers;
    }
};

/**
 * Load all sentences at once (legacy method, still used for saving)
 */
std::vector<std::string> load_sentences(const std::string& file_path) {
    std::vector<std::string> sentences;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << file_path << std::endl;
        // Generate some sample sentences if file doesn't exist
        for (int i = 0; i < 50; ++i) {
            sentences.push_back("This is a sample sentence " + std::to_string(i+1) + 
                               " for embedding. It will be processed by a worker thread.");
        }
        return sentences;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            sentences.push_back(line);
        }
    }
    
    return sentences;
}

/**
 * Save embeddings to file
 */
void save_embeddings(const std::string& output_file, 
                    const std::vector<std::string>& sentences,
                    const std::vector<torch::Tensor>& embeddings) {
    
    // In a real implementation, you would serialize the embeddings
    // For simplicity, we just print the shape and first values
    std::cout << "\nSaving " << embeddings.size() << " embeddings to " << output_file << std::endl;
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file for writing: " << output_file << std::endl;
        return;
    }
    
    file << "{\n  \"embeddings\": [\n";
    
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto& embedding = embeddings[i];
        
        // Write the text and first 5 values of embedding
        file << "    {\n";
        file << "      \"text\": \"" << sentences[i] << "\",\n";
        file << "      \"embedding\": [";
        
        // Get the first 10 values (or fewer if there are fewer)
        int limit = std::min(10, static_cast<int>(embedding.size(1)));
        auto values = embedding.slice(1, 0, limit);
        auto accessor = values.accessor<float, 2>();
        
        for (int j = 0; j < limit; ++j) {
            file << accessor[0][j];
            if (j < limit - 1) {
                file << ", ";
            }
        }
        
        file << ", ...],\n";  // Indicate there are more values
        file << "      \"dimensions\": " << embedding.size(1) << "\n";
        file << "    }";
        
        if (i < embeddings.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n}\n";
}

/**
 * Main function: Run multithreaded embeddings with progress tracking
 */
int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string model_path = "embeddings_model.pt";
        std::string input_file = "sentences.txt";
        std::string output_file = "embeddings_output.json";
        int num_threads = std::thread::hardware_concurrency();
        size_t batch_size = 10; // Default batch size
        
        // Allow overriding paths and thread count from command line
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (arg == "--input" && i + 1 < argc) {
                input_file = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_file = argv[++i];
            } else if (arg == "--threads" && i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else if (arg == "--batch-size" && i + 1 < argc) {
                batch_size = static_cast<size_t>(std::stoi(argv[++i]));
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --model PATH     Path to the model file (default: embeddings_model.pt)\n"
                          << "  --input PATH     Path to input sentences file (default: sentences.txt)\n"
                          << "  --output PATH    Path to output embeddings file (default: embeddings_output.json)\n"
                          << "  --threads N      Number of worker threads (default: " << num_threads << ")\n"
                          << "  --batch-size N   Number of sentences to load in each batch (default: 10)\n"
                          << "  --help           Show this help message\n";
                return 0;
            }
        }
        
        std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
        
        // Set device (CPU or CUDA)
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        } else {
            std::cout << "CUDA is not available. Using CPU." << std::endl;
        }
        
        // Load the model
        std::cout << "Loading model from: " << model_path << std::endl;
        auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path, device));
        model->eval();
        
        std::cout << "Model loaded successfully!" << std::endl;
        
        // Create batch loader
        auto batch_loader = std::make_shared<BatchLoader>(input_file);
        batch_loader->open();
        
        size_t total_sentences = batch_loader->get_total_count();
        std::cout << "Found " << total_sentences << " sentences in total" << std::endl;
        
        // Adjust thread count if we have few sentences
        if (total_sentences < static_cast<size_t>(num_threads)) {
            num_threads = std::max(1, static_cast<int>(total_sentences));
            std::cout << "Reducing thread count to " << num_threads 
                      << " due to small number of sentences" << std::endl;
        }
        
        // Batch size is already defined from command line parameters
        
        // Create workers
        std::vector<std::shared_ptr<EmbeddingWorker>> workers;
        for (int i = 0; i < num_threads; ++i) {
            workers.push_back(std::make_shared<EmbeddingWorker>(
                i + 1, batch_loader, batch_size, total_sentences / num_threads, model, device
            ));
        }
        
        // Create and start progress tracker
        ProgressTracker tracker;
        for (auto& worker : workers) {
            tracker.add_worker(worker);
        }
        tracker.start();
        
        // Start worker threads
        std::vector<std::thread> threads;
        for (auto& worker : workers) {
            threads.emplace_back(&EmbeddingWorker::process, worker);
        }
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        // Stop the tracker
        tracker.stop();
        
        // Collect and combine results
        std::vector<torch::Tensor> all_embeddings;
        std::vector<std::string> all_processed_sentences;
        
        // Collect results from all workers
        for (auto& worker : workers) {
            const auto& results = worker->get_results();
            all_embeddings.insert(all_embeddings.end(), results.begin(), results.end());
            
            // Also collect the processed sentences in the same order
            const auto& processed_sentences = worker->get_processed_sentences();
            all_processed_sentences.insert(all_processed_sentences.end(), 
                                          processed_sentences.begin(), 
                                          processed_sentences.end());
        }
        
        // Save embeddings to file
        save_embeddings(output_file, all_processed_sentences, all_embeddings);
        
        std::cout << "\nAll tasks completed! Embeddings saved to " << output_file << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Torch error: " << e.what() << std::endl;
        std::cerr << "This may be due to a model format mismatch or incorrect inputs." << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
