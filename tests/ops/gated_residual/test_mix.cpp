#include "mix/oracle.h"
#include "core/decode_graph.h"

namespace {
using namespace ninfer;
using namespace ninfer::test::hc::mix;

int run(Weights& weights, std::span<const float> catalog, std::span<const Column> ref, int m,
        CudaExecutionView execution, bool captured = false) {
    Fixture f(weights, m);
    f.upload(catalog, execution.stream());
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        definition.capture(execution.stream(), [&] { f.invoke(execution); });
        graph.instantiate(definition);
        graph.launch(execution.stream());
    } else
        f.invoke(execution);
    ninfer::test::cuda_synchronize(execution.stream());
    const auto label = std::string(name(weights.kind)) + " M=" + std::to_string(m) +
                       " pattern=" + std::to_string(int(weights.pattern)) +
                       " graph=" + std::to_string(captured);
    int failed = verify(f, ref, label) + f.check(label);
    if (captured) {
        const auto next     = input_catalog(weights.pattern, 1);
        const auto next_ref = reference(weights, next);
        f.upload(next, execution.stream());
        graph.launch(execution.stream());
        ninfer::test::cuda_synchronize(execution.stream());
        failed += verify(f, next_ref, label + " changed input") + f.check(label + " changed input");
    }
    return failed;
}

// A prefill scheduler can place the same normalized prefix in differently sized
// windows. Above the short-window schedules, the represented HC result stays exact.
int prefill_prefix(Weights& weights, std::span<const float> catalog, CudaExecutionView e) {
    std::vector<std::uint16_t> expected;
    int failed = 0;
    for (int m : {320, 1024, 1536, 2048, 3599}) {
        Fixture f(weights, m);
        f.upload(catalog, e.stream());
        f.invoke(e);
        ninfer::test::cuda_synchronize(e.stream());
        const auto words = ninfer::test::from_device<std::uint16_t>(
            f.output.data(), std::size_t(kWidth) * kCatalogColumns);
        if (expected.empty()) expected = words;
        else failed += ninfer::test::verify_exact("HC prefill prefix across window sizes", words, expected);
        failed += f.check("HC prefill prefix");
    }
    return failed;
}

int suite(Kind kind, CudaExecutionView e) {
    Weights weights(kind, Pattern::Dense);
    const auto catalog = input_catalog(Pattern::Dense);
    const auto ref     = reference(weights, catalog);
    int failed         = 0;
    failed += prefill_prefix(weights, catalog, e);
    for (int m :
         {1,   2,   4,    5,    16,   31,   32,   33,   47,   48,   49,   63,   64,   65,   95,   96,   97,   127,
          128, 129, 159,  160,  161,  191,  192,  193,  223,  224,  225,  255,  256,  257,  319,
          320, 321, 383,  384,  385,  447,  448,  449,  511,  512,  513,  671,  672,  673,  767,
          768, 769, 1024, 1199, 1200, 1201, 1359, 1360, 1361, 1536, 1631, 1632, 1633, 1792, 2048})
        failed += run(weights, catalog, ref, m, e);
    failed += run(weights, catalog, ref, 7 * 8, e, true);
    failed += run(weights, catalog, ref, 385, e, true);
    failed += validate(weights, ref, e);
    failed += weights.check();
    if (kind == Kind::Block) {
        Weights separate(kind, Pattern::Dense, true);
        const auto oracle = reference(separate, catalog);
        failed += prefill_prefix(separate, catalog, e);
        for (int m : {1, 2, 5, 17, 47, 48, 49, 64, 65, 128, 129, 385, 769, 2048})
            failed += run(separate, catalog, oracle, m, e);
        failed += run(separate, catalog, oracle, 17, e, true);
        failed += validate(separate, oracle, e);
        failed += separate.check();
    }
    for (Pattern pattern :
         {Pattern::Zero, Pattern::InjectionOnly, Pattern::RowMapping, Pattern::Saturation}) {
        if (kind == Kind::Final && pattern == Pattern::InjectionOnly) continue;
        Weights structured(kind, pattern);
        const auto input    = input_catalog(pattern);
        const auto expected = reference(structured, std::span{input}.first(3 * kTotal));
        failed += run(structured, std::span{input}.first(3 * kTotal), expected, 3, e);
        if (pattern == Pattern::RowMapping) {
            for (int m : {65, 385, 769})
                failed += run(structured, std::span{input}.first(3 * kTotal), expected, m, e);
        }
        failed += structured.check();
    }
    std::cout << (failed ? "FAIL " : "OK ") << name(kind) << '\n';
    return failed;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    DeviceContext device;
    const auto e     = device.execution_view();
    const int failed = suite(Kind::Block, e) + suite(Kind::Final, e);
    return failed ? 1 : 0;
}
