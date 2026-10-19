#include "runner.h"
#include "oracle.h"
#include "core/decode_graph.h"

namespace ninfer::test::gdn_control {

int verify_column_invariance(CudaExecutionView execution) {
    int failed = 0;
    // Ordinary decode, speculative widths, and prefill chunks must interpret the
    // same represented input column identically. Numerical accuracy is checked
    // independently by run() against projection_reference().
    for (auto dtype : {DType::BF16, DType::FP32}) {
        Fixture f({48, 2560, Storage::Parent, dtype}, {8193}, false);
        f.invoke(f.arena, execution);
        cuda_synchronize(execution.stream());
        const auto reference_g = from_device<std::uint32_t>(f.g.data(), 48 * 8193);
        const auto reference_b = from_device<std::uint32_t>(f.beta.data(), 48 * 8193);
        for (int columns : {1, 2, 8, 9, 16, 17, 48, 64, 65, 129, 193, 449, 1025}) {
            const auto input = f.input.slice(1, 0, columns);
            ops::GdnControlOutputs out{f.outputs.g.slice(1, 0, columns),
                                       f.outputs.beta.slice(1, 0, columns)};
            ops::gdn_control_proj(input, f.weights.prepared, f.a_log_tensor,
                                  f.dt_bias_tensor, out, f.arena, execution);
            cuda_synchronize(execution.stream());
            const auto count = std::size_t{48} * columns;
            const auto name = "Flash-Next control shared columns T=" + std::to_string(columns);
            failed += verify_exact((name + " g").c_str(),
                from_device<std::uint32_t>(f.g.data(), count),
                std::vector<std::uint32_t>(reference_g.begin(), reference_g.begin() + count));
            failed += verify_exact((name + " beta").c_str(),
                from_device<std::uint32_t>(f.beta.data(), count),
                std::vector<std::uint32_t>(reference_b.begin(), reference_b.begin() + count));
        }
    }
    return failed;
}

int run(Profile p, Case c, bool normalize_input, CudaExecutionView execution) {
    Fixture f(p, c, normalize_input);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const bool captured = c.execution == Execution::Graph;
    const auto launch   = [&] { f.invoke(f.arena, execution); };
    if (captured) {
        launch();
        cuda_synchronize(execution.stream());
        definition.capture(execution.stream(), launch);
        graph.instantiate(definition);
    }
    int failed = 0;
    for (int phase = 0; phase < (captured ? 3 : 1); ++phase) {
        if (phase) f.advance();
        f.repaint(execution.stream());
        if (captured)
            graph.launch(execution.stream());
        else
            launch();
        cuda_synchronize(execution.stream());
        const auto name = label(p, c, normalize_input) + " phase=" + std::to_string(phase);
        failed += verify_projection(name, f, projection_reference(f));
        failed += f.check(name);
        const auto peak = f.capacity == 0 ? f.prefix : f.capacity + (f.prefix ? 256 : 0);
        if (f.arena.used() != f.prefix || f.arena.peak_used() != peak) {
            std::cerr << name << ": arena restoration/capacity mismatch\n";
            ++failed;
        }
    }
    return failed;
}

} // namespace ninfer::test::gdn_control
