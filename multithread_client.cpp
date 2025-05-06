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


#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"


const std::vector<std::string> COLORS = {GREEN, YELLOW, BLUE, MAGENTA, CYAN};


class ProgressBar;
class EmbeddingWorker;
class ProgressTracker;


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
        
        float percent = static_cast<float>(value) / static_cast<float>(total) * 100.0f;
        
        
        int filled_length = static_cast<int>(length * value / total);
        
        
        std::string bar;
        for (int i = 0; i < length; ++i) {
            if (i < filled_length) {
                bar += fill;
            } else {
                bar += '-';
            }
        }
        
        
        std::ostringstream percent_stream;
        percent_stream << std::fixed << std::setprecision(decimals) << percent;
        std::string percent_str = percent_stream.str();
        
        
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
 * Worker class that handles embedding a set of sentences
 */
class EmbeddingWorker {
private:
    int task_id;
    std::vector<std::string> sentences;
    std::shared_ptr<torch::jit::script::Module> model;
    torch::Device device;
    std::unique_ptr<ProgressBar> progress_bar;
    std::atomic<bool> is_done;
    std::vector<torch::Tensor> results;
    
public:
    EmbeddingWorker(int task_id, 
                   const std::vector<std::string>& sentences,
                   std::shared_ptr<torch::jit::script::Module> model,
                   torch::Device device)
        : task_id(task_id), 
          sentences(sentences), 
          model(model), 
          device(device), 
          is_done(false) {
              
        
        std::string color = COLORS[task_id % COLORS.size()];
        progress_bar = std::make_unique<ProgressBar>(
            sentences.size(),
            "Thread " + std::to_string(task_id),
            "Complete",
            1, 30, '#', color
        );
        
        
        results.reserve(sentences.size());
    }
    
    
    std::pair<torch::Tensor, torch::Tensor> tokenize(const std::string& text) {
        
        
        
        
        
        int seq_len = 10 + (text.length() % 10); 
        
        
        auto input_ids = torch::ones({1, seq_len}, torch::kInt64).to(device);
        
        
        auto attention_mask = torch::ones({1, seq_len}, torch::kInt64).to(device);
        
        return {input_ids, attention_mask};
    }
    
    
    void process() {
        size_t count = 0;
        
        for (const auto& sentence : sentences) {
            
            auto [input_ids, attention_mask] = tokenize(sentence);
            
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_ids);
            inputs.push_back(attention_mask);
            
            
            torch::NoGradGuard no_grad;
            torch::Tensor embedding = model->forward(inputs).toTensor();
            
            
            results.push_back(embedding);
            
            
            count++;
            progress_bar->update(count);
            
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50 + (std::rand() % 200)));
        }
        
        
        is_done = true;
    }
    
    
    bool done() const { return is_done; }
    const ProgressBar& get_progress_bar() const { return *progress_bar; }
    const std::vector<torch::Tensor>& get_results() const { return results; }
    int get_id() const { return task_id; }
};


class ProgressTracker {
private:
    std::vector<std::shared_ptr<EmbeddingWorker>> workers;
    std::atomic<bool> stop_flag;
    std::thread display_thread;
    
    void display_loop() {
        
        std::cout << "\033[2J\033[H";
        
        while (!stop_flag) {
            
            std::cout << "\033[H";
            
            
            size_t total_done = 0;
            size_t total_work = 0;
            
            
            for (const auto& worker : workers) {
                const auto& progress_bar = worker->get_progress_bar();
                std::cout << progress_bar.print() << std::endl;
                
                
                total_done += progress_bar.get_value();
                total_work += progress_bar.get_total();
            }
            
            
            float overall_percent = (total_work > 0) ? 
                (static_cast<float>(total_done) / total_work * 100.0f) : 0.0f;
            
            std::cout << "\nOverall progress: " 
                      << std::fixed << std::setprecision(1) 
                      << overall_percent << "%" << std::endl;
            
            
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
            
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        
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


std::vector<std::string> load_sentences(const std::string& file_path) {
    std::vector<std::string> sentences;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << file_path << std::endl;
        
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


std::vector<std::vector<std::string>> distribute_sentences(
    const std::vector<std::string>& sentences, int num_workers) {
    
    std::vector<std::vector<std::string>> distributed(num_workers);
    
    for (size_t i = 0; i < sentences.size(); ++i) {
        distributed[i % num_workers].push_back(sentences[i]);
    }
    
    return distributed;
}


void save_embeddings(const std::string& output_file, 
                    const std::vector<std::string>& sentences,
                    const std::vector<torch::Tensor>& embeddings) {
    
    
    
    std::cout << "\nSaving " << embeddings.size() << " embeddings to " << output_file << std::endl;
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file for writing: " << output_file << std::endl;
        return;
    }
    
    file << "{\n  \"embeddings\": [\n";
    
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto& embedding = embeddings[i];
        
        
        file << "    {\n";
        file << "      \"text\": \"" << sentences[i] << "\",\n";
        file << "      \"embedding\": [";
        
        
        int limit = std::min(10, static_cast<int>(embedding.size(1)));
        auto values = embedding.slice(1, 0, limit);
        auto accessor = values.accessor<float, 2>();
        
        for (int j = 0; j < limit; ++j) {
            file << accessor[0][j];
            if (j < limit - 1) {
                file << ", ";
            }
        }
        
        file << ", ...],\n";  
        file << "      \"dimensions\": " << embedding.size(1) << "\n";
        file << "    }";
        
        if (i < embeddings.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n}\n";
}

int main(int argc, char* argv[]) {
    try {
        
        std::string model_path = "embeddings_model.pt";
        std::string input_file = "sentences.txt";
        std::string output_file = "embeddings_output.json";
        int num_threads = std::thread::hardware_concurrency();
        
        
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
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  --model PATH     Path to the model file (default: embeddings_model.pt)\n"
                          << "  --input PATH     Path to input sentences file (default: sentences.txt)\n"
                          << "  --output PATH    Path to output embeddings file (default: embeddings_output.json)\n"
                          << "  --threads N      Number of worker threads (default: " << num_threads << ")\n"
                          << "  --help           Show this help message\n";
                return 0;
            }
        }
        
        std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
        
        
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available! Using GPU." << std::endl;
            device = torch::Device(torch::kCUDA);
        } else {
            std::cout << "CUDA is not available. Using CPU." << std::endl;
        }
        
        
        std::cout << "Loading model from: " << model_path << std::endl;
        auto model = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path, device));
        model->eval();
        
        std::cout << "Model loaded successfully!" << std::endl;
        
        
        std::vector<std::string> sentences = load_sentences(input_file);
        std::cout << "Loaded " << sentences.size() << " sentences" << std::endl;
        
        
        if (sentences.size() < static_cast<size_t>(num_threads)) {
            num_threads = std::max(1, static_cast<int>(sentences.size()));
            std::cout << "Reducing thread count to " << num_threads 
                      << " due to small number of sentences" << std::endl;
        }
        
        
        auto distributed_sentences = distribute_sentences(sentences, num_threads);
        
        
        std::vector<std::shared_ptr<EmbeddingWorker>> workers;
        for (int i = 0; i < num_threads; ++i) {
            workers.push_back(std::make_shared<EmbeddingWorker>(
                i + 1, distributed_sentences[i], model, device
            ));
        }
        
        
        ProgressTracker tracker;
        for (auto& worker : workers) {
            tracker.add_worker(worker);
        }
        tracker.start();
        
        
        std::vector<std::thread> threads;
        for (auto& worker : workers) {
            threads.emplace_back(&EmbeddingWorker::process, worker);
        }
        
        
        for (auto& t : threads) {
            t.join();
        }
        
        
        tracker.stop();
        
        
        std::vector<torch::Tensor> all_embeddings;
        for (auto& worker : workers) {
            const auto& results = worker->get_results();
            all_embeddings.insert(all_embeddings.end(), results.begin(), results.end());
        }
        
        
        save_embeddings(output_file, sentences, all_embeddings);
        
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
