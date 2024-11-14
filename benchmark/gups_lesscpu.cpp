#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <pthread.h>

class NumaMemoryManager {
public:
    static void bind_to_cpu_node(int cpu_node) {
        if (numa_available() < 0) {
            throw std::runtime_error("NUMA not available");
        }

        // 设置CPU亲和性
        bitmask* cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(cpu_node, cpu_mask);
        if (set_sched_affinity(0, cpu_mask) != 0) {
            numa_free_cpumask(cpu_mask);
            throw std::runtime_error("Failed to set CPU affinity");
        }
        numa_free_cpumask(cpu_mask);
    }

    static void* allocate_on_node(size_t size, int memory_node) {
        // 分配内存
        void* ptr = numa_alloc_onnode(size, memory_node);
        if (!ptr) {
            throw std::runtime_error("Failed to allocate memory on specified NUMA node");
        }
        
        // 确保内存实际分配到指定节点
        unsigned long nodemask = 1UL << memory_node;
        if (mbind(ptr, size, MPOL_BIND, &nodemask, sizeof(unsigned long) * 8, MPOL_MF_MOVE) != 0) {
            numa_free(ptr, size);
            throw std::runtime_error("Failed to bind memory to NUMA node");
        }

        return ptr;
    }

    static void free_numa_memory(void* ptr, size_t size) {
        if (ptr) {
            numa_free(ptr, size);
        }
    }
};

class CPUlessNumaGUPS {
private:
    struct ThreadContext {
        int cpu_node;          // 运行线程的CPU所在节点
        int memory_node;       // 访问内存的目标节点
        size_t updates_count;  // 需要执行的更新次数
        size_t chunk_size;     // 每个线程负责的内存块大小
        void* memory_area;     // 线程操作的内存区域
    };

    std::vector<ThreadContext> thread_contexts;
    std::atomic<uint64_t> updates_completed{0};
    size_t total_memory_size;
    int num_threads;
    size_t updates_per_thread;

    static void pin_thread_to_cpu_node(int cpu_node) {
        bitmask* cpu_mask = numa_allocate_cpumask();
        numa_node_to_cpus(cpu_node, cpu_mask);
        sched_setaffinity(0, numa_bitmask_nbytes(cpu_mask), (cpu_set_t*)cpu_mask);
        numa_free_cpumask(cpu_mask);
    }

    void worker_thread(ThreadContext& ctx) {
        // 将线程绑定到指定的CPU节点
        pin_thread_to_cpu_node(ctx.cpu_node);

        // 初始化随机数生成器
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, ctx.chunk_size - sizeof(uint64_t));

        // 执行随机更新
        for (size_t i = 0; i < ctx.updates_count; i++) {
            size_t offset = dis(gen);
            uint64_t* target = (uint64_t*)((char*)ctx.memory_area + offset);
            
            // 读-修改-写操作
            uint64_t value = *target;
            *target = value ^ gen();

            updates_completed.fetch_add(1, std::memory_order_relaxed);
        }
    }

public:
    struct Config {
        size_t total_memory_size;  // 总内存大小
        int memory_node;           // CXL内存所在的NUMA节点
        std::vector<int> cpu_nodes; // 可用的CPU节点列表
        size_t updates_per_thread;  // 每个线程的更新次数
    };

    CPUlessNumaGUPS(const Config& config) 
        : total_memory_size(config.total_memory_size)
        , num_threads(config.cpu_nodes.size())
        , updates_per_thread(config.updates_per_thread) {
        
        // 检查NUMA可用性
        if (numa_available() < 0) {
            throw std::runtime_error("NUMA not available");
        }

        // 为每个线程创建上下文
        size_t chunk_size = total_memory_size / num_threads;
        for (int i = 0; i < num_threads; i++) {
            ThreadContext ctx;
            ctx.cpu_node = config.cpu_nodes[i];
            ctx.memory_node = config.memory_node;
            ctx.updates_count = updates_per_thread;
            ctx.chunk_size = chunk_size;
            
            // 在指定NUMA节点上分配内存
            ctx.memory_area = NumaMemoryManager::allocate_on_node(chunk_size, config.memory_node);
            
            thread_contexts.push_back(ctx);
        }
    }

    ~CPUlessNumaGUPS() {
        // 清理分配的内存
        for (auto& ctx : thread_contexts) {
            NumaMemoryManager::free_numa_memory(ctx.memory_area, ctx.chunk_size);
        }
    }

    double run_benchmark() {
        std::vector<std::thread> threads;
        updates_completed.store(0);

        auto start = std::chrono::high_resolution_clock::now();

        // 启动工作线程
        for (auto& ctx : thread_contexts) {
            threads.emplace_back(&CPUlessNumaGUPS::worker_thread, this, std::ref(ctx));
        }

        // 监控进度
        size_t total_updates = num_threads * updates_per_thread;
        while (updates_completed.load(std::memory_order_relaxed) < total_updates) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        
        return (total_updates / seconds) / 1e9; // 返回GUPS值
    }
};

int main() {
    try {
        // 获取系统NUMA拓扑信息
        int num_nodes = numa_num_configured_nodes();
        std::vector<int> cpu_nodes;
        
        // 找出所有有CPU的节点
        for (int i = 0; i < num_nodes; i++) {
            bitmask* cpu_mask = numa_allocate_cpumask();
            numa_node_to_cpus(i, cpu_mask);
            if (numa_bitmask_weight(cpu_mask) > 0) {
                cpu_nodes.push_back(i);
            }
            numa_free_cpumask(cpu_mask);
        }

        // 配置benchmark
        CPUlessNumaGUPS::Config config {
            .total_memory_size = 4ULL * 1024 * 1024 * 1024,  // 4GB
            .memory_node = 1,  // CXL内存节点
            .cpu_nodes = cpu_nodes,  // 所有有CPU的节点
            .updates_per_thread = 10000000  // 每线程更新次数
        };

        std::cout << "Configuration:\n"
                  << "Memory Node: " << config.memory_node << "\n"
                  << "Available CPU Nodes: ";
        for (int node : config.cpu_nodes) {
            std::cout << node << " ";
        }
        std::cout << "\n"
                  << "Total Memory: " << (config.total_memory_size / (1024*1024)) << " MB\n"
                  << "Updates per Thread: " << config.updates_per_thread << "\n\n";

        CPUlessNumaGUPS gups(config);
        double gups_value = gups.run_benchmark();
        
        std::cout << "\nResults:\n"
                  << "GUPS: " << gups_value << "\n"
                  << "Total Updates: " << (config.cpu_nodes.size() * config.updates_per_thread) 
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}