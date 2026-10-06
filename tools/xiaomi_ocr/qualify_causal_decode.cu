#include "ninfer/ops/softmax_attention.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ninfer;
namespace {
constexpr int D = 256, HQ = 8, HK = 2;
struct Problem {
    int visible, width, batch;
    bool masked = false;
    int envelope_max = 0;
    float sigma = .4f;
};
void upload(DeviceBuffer& destination, const auto& source) {
    destination.copy_from_host(source.data(), source.size() * sizeof(source[0]));
}
class Fixture {
public:
    Problem p;
    int capacity, pages, physical_pages;
    std::vector<__nv_bfloat16> query, key, value, expected_key;
    std::vector<__half> expected_value;
    std::vector<int> positions, tables, table_rows, valid;
    DeviceBuffer q, k, v, pos, table, rows, mask, cache_key, cache_value, output;
    ops::AttentionHeadGeometry geometry{D, HQ, HK};
    ops::CausalAttentionExecutionEnvelope envelope;
    WorkspaceArena workspace;

    explicit Fixture(Problem problem)
        : p(problem), capacity(p.envelope_max ? p.envelope_max : p.visible),
          pages((capacity + 63) / 64), physical_pages(pages * p.batch),
          query(D * HQ * p.width * p.batch), key(D * HK * p.width * p.batch), value(key.size()),
          expected_key(std::size_t(D) * 64 * HK * physical_pages), expected_value(expected_key.size()),
          positions(p.width * p.batch), tables(physical_pages), table_rows(p.batch), valid(p.batch, p.width),
          q(query.size() * 2), k(key.size() * 2), v(value.size() * 2), pos(positions.size() * 4),
          table(tables.size() * 4), rows(table_rows.size() * 4), mask(valid.size() * 4),
          cache_key(expected_key.size() * 2), cache_value(cache_key.bytes), output(q.bytes),
          envelope{1, static_cast<unsigned>(capacity)},
          workspace(ops::causal_softmax_attention_workspace_capacity_bytes(
              geometry, KvCacheStorage::BFloat16, envelope, p.batch, p.width, p.width) + 256) {
        std::mt19937 random(781239 + p.visible + p.width * 19 + p.batch * 47);
        std::normal_distribution<float> distribution(0, p.sigma);
        for (auto& x : query) x = __float2bfloat16(distribution(random));
        for (auto& x : expected_key) x = __float2bfloat16(distribution(random));
        for (auto& x : expected_value) x = __float2half(__bfloat162float(__float2bfloat16(distribution(random))));
        for (int b = 0; b < p.batch; ++b) {
            table_rows[b] = p.batch - b - 1;
            for (int page = 0; page < pages; ++page)
                tables[table_rows[b] * pages + page] = physical_pages - 1 - (b * pages + page);
            if (p.masked) valid[b] = b == 1 ? 0 : b == 2 ? std::max(1, p.width - 1) : p.width;
            for (int t = 0; t < p.width; ++t) {
                positions[b * p.width + t] = t < valid[b] ? p.visible - p.width + t
                    : valid[b] ? p.visible - p.width + valid[b] - 1 : 0;
                for (int h = 0; h < HK; ++h)
                    for (int d = 0; d < D; ++d) {
                        const int input = ((b * p.width + t) * HK + h) * D + d;
                        key[input] = __float2bfloat16(distribution(random));
                        value[input] = __float2bfloat16(distribution(random));
                        if (t < valid[b]) {
                            const auto index = address(b, positions[b * p.width + t], h, d);
                            expected_key[index] = key[input];
                            expected_value[index] = __float2half(__bfloat162float(value[input]));
                        }
                    }
            }
        }
        upload(q, query); upload(k, key); upload(v, value); upload(pos, positions);
        upload(table, tables); upload(rows, table_rows); upload(mask, valid);
        reset_cache();
    }
    std::size_t address(int batch, int logical_key, int head, int feature) const {
        const int page = tables[table_rows[batch] * pages + logical_key / 64];
        return ((std::size_t(page) * HK + head) * 64 + logical_key % 64) * D + feature;
    }
    PagedKVBatchLayerView view(void* key_pointer = nullptr, void* value_pointer = nullptr) {
        return {.k_pages = Tensor(key_pointer ? key_pointer : cache_key.p, DType::BF16, {D,64,HK,physical_pages}),
                .v_pages = Tensor(value_pointer ? value_pointer : cache_value.p, DType::FP16, {D,64,HK,physical_pages}),
                .block_tables = Tensor(table.p, DType::I32, {pages,p.batch}),
                .head_dim = D, .num_kv_heads = HK, .storage = KvCacheStorage::BFloat16};
    }
    void reset_cache() {
        auto initial_key = expected_key;
        auto initial_value = expected_value;
        for (int b = 0; b < p.batch; ++b)
            for (int t = 0; t < valid[b]; ++t)
                for (int h = 0; h < HK; ++h)
                    for (int d = 0; d < D; ++d) {
                        const auto index = address(b, positions[b * p.width + t], h, d);
                        initial_key[index] = __float2bfloat16(0);
                        initial_value[index] = __float2half(0);
                    }
        upload(cache_key, initial_key); upload(cache_value, initial_value);
        CUDA_CHECK(cudaMemset(output.p, 0xff, output.bytes));
    }
    void run(cudaStream_t stream, void* key_pointer = nullptr, void* value_pointer = nullptr) {
        workspace.reset();
        Tensor result(output.p, DType::BF16, {D,HQ,p.width,p.batch});
        ops::causal_softmax_attention(Tensor(q.p, DType::BF16, {D,HQ,p.width,p.batch}),
            Tensor(k.p, DType::BF16, {D,HK,p.width,p.batch}), Tensor(v.p, DType::BF16, {D,HK,p.width,p.batch}),
            Tensor(pos.p, DType::I32, {p.width,p.batch}),
            p.masked ? Tensor(mask.p, DType::I32, {p.batch}) : Tensor{}, Tensor(rows.p, DType::I32, {p.batch}),
            geometry, .0625f, view(key_pointer, value_pointer), envelope, workspace,
            result, stream);
    }
    std::vector<double> oracle() const {
        std::vector<double> result(query.size());
        for (int b = 0; b < p.batch; ++b)
            for (int t = 0; t < valid[b]; ++t)
                for (int h = 0; h < HQ; ++h) {
                    const int count = positions[b * p.width + t] + 1;
                    const int qb = ((b * p.width + t) * HQ + h) * D;
                    std::vector<double> scores(count);
                    double maximum = -1e300;
                    for (int j = 0; j < count; ++j) {
                        double dot = 0;
                        const auto base = address(b, j, h / (HQ / HK), 0);
                        for (int d = 0; d < D; ++d)
                            dot += double(__bfloat162float(query[qb + d])) * __bfloat162float(expected_key[base + d]);
                        scores[j] = dot / 16.; maximum = std::max(maximum, scores[j]);
                    }
                    std::array<double, D> sum{};
                    double denominator = 0;
                    for (int j = 0; j < count; ++j) {
                        const double weight = std::exp(scores[j] - maximum);
                        denominator += weight;
                        const auto base = address(b, j, h / (HQ / HK), 0);
                        for (int d = 0; d < D; ++d) sum[d] += weight * double(__half2float(expected_value[base + d]));
                    }
                    for (int d = 0; d < D; ++d) result[qb + d] = sum[d] / denominator;
                }
        return result;
    }
    bool compare(const std::vector<double>& ideal, bool cached) {
        std::vector<__nv_bfloat16> actual(query.size()), actual_key(expected_key.size());
        std::vector<__half> actual_value(expected_value.size());
        output.copy_to_host(actual.data(), output.bytes);
        cache_key.copy_to_host(actual_key.data(), cache_key.bytes);
        cache_value.copy_to_host(actual_value.data(), cache_value.bytes);
        double ss = 0, dd = 0, maximum = 0;
        bool finite = true, tails_zero = true;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double value = __bfloat162float(actual[i]);
            finite = finite && std::isfinite(value);
            const int b = int(i) / (p.width * HQ * D), t = (int(i) / (HQ * D)) % p.width;
            if (t >= valid[b]) tails_zero = tails_zero && value == 0;
            const double delta = value - ideal[i];
            ss += ideal[i] * ideal[i]; dd += delta * delta; maximum = std::max(maximum, std::abs(delta));
        }
        const double nrms = std::sqrt(dd / ss), peak = maximum / std::sqrt(ss / actual.size());
        const bool cache_exact = std::memcmp(expected_key.data(), actual_key.data(), cache_key.bytes) == 0
            && std::memcmp(expected_value.data(), actual_value.data(), cache_value.bytes) == 0;
        const bool pass = finite && tails_zero && nrms < .006 && peak < .045 && cache_exact;
        std::cout << "{\"kind\":\"qualification\",\"visible\":" << p.visible << ",\"envelope_max\":" << capacity
            << ",\"width\":" << p.width << ",\"batch\":" << p.batch << ",\"masked\":" << (p.masked ? "true" : "false")
            << ",\"cached\":" << (cached ? "true" : "false") << ",\"sigma\":" << p.sigma
            << ",\"nrms\":" << nrms << ",\"peak\":" << peak << ",\"tails_zero\":" << (tails_zero ? "true" : "false")
            << ",\"cache_exact\":" << (cache_exact ? "true" : "false") << ",\"pass\":" << (pass ? "true" : "false") << "}\n" << std::flush;
        return pass;
    }
    bool qualify() {
        const auto ideal = oracle();
        run(nullptr); CUDA_CHECK(cudaDeviceSynchronize());
        bool pass = compare(ideal, false);
        if (p.batch == 1 && !p.masked) {
            const PagedKVLayerView single{.k_pages = view().k_pages, .v_pages = view().v_pages,
                .block_table = Tensor(table.p, DType::I32, {pages}), .head_dim = D,
                .num_kv_heads = HK, .storage = KvCacheStorage::BFloat16};
            workspace.reset(); CUDA_CHECK(cudaMemset(output.p, 0xff, output.bytes));
            Tensor result(output.p, DType::BF16, {D,HQ,p.width});
            ops::causal_softmax_attention_cached(Tensor(q.p, DType::BF16, {D,HQ,p.width}),
                Tensor(pos.p, DType::I32, {p.width}), geometry, .0625f, single, envelope,
                workspace, result, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            pass = compare(ideal, true) && pass;
        }
        return pass;
    }
    void benchmark() {
        const auto cache_bytes = cache_key.bytes + cache_value.bytes;
        const int replicas = int((128u * 1024 * 1024 + cache_bytes - 1) / cache_bytes);
        const int graph_repetitions = std::max(1, (64 + replicas - 1) / replicas);
        const int calls_per_graph = replicas * graph_repetitions;
        DeviceBuffer key_pool(cache_key.bytes * replicas), value_pool(cache_value.bytes * replicas);
        for (int r = 0; r < replicas; ++r) {
            CUDA_CHECK(cudaMemcpy(static_cast<char*>(key_pool.p) + r * cache_key.bytes,
                expected_key.data(), cache_key.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(static_cast<char*>(value_pool.p) + r * cache_value.bytes,
                expected_value.data(), cache_value.bytes, cudaMemcpyHostToDevice));
        }
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph; cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int repetition = 0; repetition < graph_repetitions; ++repetition)
            for (int r = 0; r < replicas; ++r)
                run(stream, static_cast<char*>(key_pool.p) + r * cache_key.bytes,
                            static_cast<char*>(value_pool.p) + r * cache_value.bytes);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        cudaEvent_t begin, end; CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
        // Keep device clocks and page residency warm before recording short operator calls.
        float warmed_ms = 0, graph_ms = 0;
        do {
            CUDA_CHECK(cudaEventRecord(begin, stream)); CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaEventRecord(end, stream)); CUDA_CHECK(cudaEventSynchronize(end));
            CUDA_CHECK(cudaEventElapsedTime(&graph_ms, begin, end)); warmed_ms += graph_ms;
        } while (warmed_ms < 100);
        const int launches_per_sample = std::max(1, int(std::ceil(20.f / graph_ms)));
        std::vector<float> times;
        for (int repeat = 0; repeat < 6; ++repeat) {
            CUDA_CHECK(cudaEventRecord(begin, stream));
            for (int launch = 0; launch < launches_per_sample; ++launch)
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaEventRecord(end, stream)); CUDA_CHECK(cudaEventSynchronize(end));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
            times.push_back(ms * 1000 / (calls_per_graph * launches_per_sample));
        }
        std::sort(times.begin(), times.end());
        std::cout << "{\"kind\":\"public_op_timing\",\"visible\":" << p.visible << ",\"width\":" << p.width
            << ",\"batch\":" << p.batch << ",\"cache_pool_bytes\":" << key_pool.bytes + value_pool.bytes
            << ",\"calls_per_graph\":" << calls_per_graph
            << ",\"warmup_gpu_ms\":" << warmed_ms << ",\"graph_launches_per_sample\":" << launches_per_sample
            << ",\"workspace_capacity_bytes\":" << ops::causal_softmax_attention_workspace_capacity_bytes(
                geometry, KvCacheStorage::BFloat16, envelope, p.batch, p.width, p.width)
            << ",\"us\":" << (times[2] + times[3]) / 2 << "}\n" << std::flush;
        CUDA_CHECK(cudaEventDestroy(begin)); CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaGraphExecDestroy(executable)); CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--benchmark") {
            for (int visible : {1807, 4096, 8192, 32768})
                for (int batch : {1, 2, 4}) Fixture({visible, 1, batch}).benchmark();
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("Usage: xiaomi-ocr-qualify-causal [--benchmark]");
        bool pass = true;
        for (Problem problem : std::vector<Problem>{{65,1,1}, {513,1,4}, {1807,1,2}, {1807,1,4},
            {2048,1,1}, {2049,1,1},
            {2048,2,1}, {2048,2,4}, {1807,3,4}, {4096,4,1}, {4096,4,4,true},
            {1024,4,8,true,0,1.2f}, {513,5,1}, {513,6,4}, {513,16,1}, {513,16,4,true},
            {8192,1,1}, {8192,1,4}, {8193,1,1}, {4096,1,1,false,16384,1.2f}, {32768,1,1,false,0,1.2f}})
            pass = Fixture(problem).qualify() && pass;
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
