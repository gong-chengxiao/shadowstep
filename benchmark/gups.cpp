#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <numa.h>
#include <cstring>
#include <sstream>

struct GUPSConfig {
    size_t working_set_size;  // 总工作集大小（字节）
    size_t hot_set_size;      // 热点集大小（字节）
    size_t num_threads;       // 线程数
    int mem_node;           // Memory NUMA节点
    int cpu_node;           // CPU NUMA节点
    size_t updates_per_thread; // 每个线程的更新次数
    size_t update_object_size; // 每次更新的对象大小（字节）
    
    std::string to_string() const {
        std::stringstream ss;
        ss << "GUPS Configuration:\n"
           << "Working Set Size: " << (working_set_size / (1024*1024)) << " MB\n"
           << "Hot Set Size: " << (hot_set_size / (1024*1024)) << " MB\n"
           << "Number of Threads: " << num_threads << "\n"
           << "Memory Node: " << mem_node << "\n"
           << "CPU Node: " << cpu_node << "\n"
           << "Updates per Thread: " << updates_per_thread << "\n"
           << "Update Object Size: " << update_object_size << " bytes\n";
        return ss.str();
    }
};

class GUPS {
private:
    std::vector<char> memory_pool;     // 主内存池
    std::vector<char> hot_memory_pool; // 热点内存池
    GUPSConfig config;
    std::atomic<uint64_t> updates_completed{0};

    static void pin_thread_to_cpu_node(int cpu_node) {
        bitmask* cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(cpu_node, cpu_mask);
        sched_setaffinity(0, numa_bitmask_nbytes(cpu_mask), (cpu_set_t*)cpu_mask);
        numa_free_cpumask(cpu_mask);
    }
    
    // 生成随机地址
    uint64_t generate_random_address(uint64_t& seed, bool access_hot_set) {
        seed = (seed << 32) | (seed >> 32);
        seed *= 0x4a39b70d;
        seed ^= (seed >> 32);
        
        if (access_hot_set) {
            return (seed % (config.hot_set_size - config.update_object_size));
        } else {
            return (seed % (config.working_set_size - config.update_object_size));
        }
    }

    // 更新指定地址的数据
    void update_memory(char* base, uint64_t offset, uint64_t seed) {
        char* target = base + offset;
        for (size_t i = 0; i < config.update_object_size; i++) {
            target[i] ^= (seed + i) & 0xFF;
        }
    }

public:
    GUPS(const GUPSConfig& cfg) : config(cfg) {
        // 验证配置
        if (config.hot_set_size > config.working_set_size) {
            throw std::runtime_error("Hot set size cannot be larger than working set size");
        }
        if (config.update_object_size > config.hot_set_size) {
            throw std::runtime_error("Update object size cannot be larger than hot set size");
        }
        
        // 设置NUMA策略
        if (numa_available() >= 0) {
            numa_set_preferred(config.mem_node);
        }
        
        // 分配内存
        memory_pool.resize(config.working_set_size);
        hot_memory_pool.resize(config.hot_set_size);
        
        // 初始化内存
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;
        
        for (size_t i = 0; i < config.working_set_size; i++) {
            memory_pool[i] = dis(gen) & 0xFF;
        }
        for (size_t i = 0; i < config.hot_set_size; i++) {
            hot_memory_pool[i] = dis(gen) & 0xFF;
        }
    }

    void worker(int thread_id, double hot_set_probability) {
        // 将线程绑定到指定的CPU节点
        pin_thread_to_cpu_node(config.cpu_node);

        if (numa_available() >= 0) {
            numa_set_preferred(config.mem_node);
        }

        uint64_t seed = std::random_device{}() + thread_id;
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (size_t i = 0; i < config.updates_per_thread; i++) {
            bool access_hot_set = (dis(gen) < hot_set_probability);
            uint64_t offset = generate_random_address(seed, access_hot_set);
            
            if (access_hot_set) {
                update_memory(hot_memory_pool.data(), offset, seed);
            } else {
                update_memory(memory_pool.data(), offset, seed);
            }
            
            updates_completed.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double run_benchmark(double hot_set_probability = 0.8) {
        std::vector<std::thread> threads;
        updates_completed.store(0);

        auto start = std::chrono::high_resolution_clock::now();

        // 启动所有线程
        for (size_t i = 0; i < config.num_threads; i++) {
            threads.emplace_back(&GUPS::worker, this, i, hot_set_probability);
        }

        // 监控进度
        size_t total_updates = config.updates_per_thread * config.num_threads;
        while (updates_completed.load(std::memory_order_relaxed) < total_updates) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // 打印进度
            double progress = 100.0 * updates_completed.load() / total_updates;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) 
                      << progress << "%" << std::flush;
        }
        std::cout << std::endl;

        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        // 计算性能指标
        double total_data = total_updates * config.update_object_size;
        double bandwidth = total_data / seconds / (1024*1024*1024); // GB/s
        double gups = total_updates / seconds / 1e9;  // GUPS
        
        return gups;
    }

    void print_performance_stats(double gups) {
        std::cout << "\nPerformance Statistics:\n"
                  << "GUPS: " << gups << "\n"
                  << "Total Memory: " << (config.working_set_size / (1024*1024)) << " MB\n"
                  << "Hot Set Memory: " << (config.hot_set_size / (1024*1024)) << " MB\n"
                  << "Total Updates: " << (config.updates_per_thread * config.num_threads) << "\n"
                  << "Updates per Thread: " << config.updates_per_thread << "\n"
                  << "Update Object Size: " << config.update_object_size << " bytes\n"
                  << "NUMA Node: " << config.mem_node << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // 默认配置
    GUPSConfig config {
        .working_set_size = 96ULL * 1024 * 1024 * 1024,  // 96GB
        .hot_set_size = 32ULL * 1024 * 1024 * 1024,      // 32GB
        .num_threads = 32,
        .mem_node = 4,
        .cpu_node = 0,
        .updates_per_thread = 10000000,  // 1千万次每线程
        .update_object_size = 8         // 64字节
    };

    // 解析命令行参数
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        std::string arg = argv[i];
        std::string value = argv[i + 1];

        if (arg == "--wss") config.working_set_size = std::stoull(value) * 1024 * 1024;
        else if (arg == "--hot") config.hot_set_size = std::stoull(value) * 1024 * 1024;
        else if (arg == "--threads") config.num_threads = std::stoull(value);
        else if (arg == "--mem") config.mem_node = std::stoi(value);
        else if (arg == "--cpu") config.cpu_node = std::stoi(value);
        else if (arg == "--updates") config.updates_per_thread = std::stoull(value);
        else if (arg == "--objsize") config.update_object_size = std::stoull(value);
    }

    try {
        std::cout << config.to_string() << std::endl;
        
        GUPS gups(config);
        double gups_value = gups.run_benchmark(0.8); // 80%的访问针对热点集
        gups.print_performance_stats(gups_value);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}