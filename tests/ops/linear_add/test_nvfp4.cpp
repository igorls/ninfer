#include "ops/linear_add/linear_add_test_common.h"
#include "ops/linear_common.h"

#include <array>
#include <cstdint>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear_add;

struct Profile {
    std::int32_t n;
    std::int32_t k;
    std::int32_t first_a4;
    std::uint32_t seed;
};

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9U) ^
                                  (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_residual(std::int32_t rows, std::int32_t tokens,
                                         std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::uint32_t coordinate = static_cast<std::uint32_t>(row) * 23U +
                                             static_cast<std::uint32_t>(token) * 41U + seed * 7U;
            const float represented =
                static_cast<float>(static_cast<int>(coordinate & 0xffU) - 128) * (1.0F / 128.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

int verify_preserved(const GuardedDeviceBuffer& device, std::span<const std::uint8_t> expected,
                     std::string_view label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": payload was modified\n";
    return 1;
}

int run_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    const std::int32_t first_a4 = k == 6144 ? 17 : 9;
    const std::array invocations{
        Invocation{1, ops::LinearPolicy::A16Only},
        Invocation{4, ops::LinearPolicy::A16Only},
        Invocation{5, ops::LinearPolicy::A16Only},
        Invocation{8, ops::LinearPolicy::A16Only},
        Invocation{16, ops::LinearPolicy::A16Only},
        Invocation{17, ops::LinearPolicy::A16Only},
        Invocation{24, ops::LinearPolicy::A16Only},
        Invocation{25, ops::LinearPolicy::A16Only},
        Invocation{32, ops::LinearPolicy::A16Only},
        Invocation{33, ops::LinearPolicy::A16Only},
        Invocation{48, ops::LinearPolicy::A16Only},
        Invocation{49, ops::LinearPolicy::A16Only},
        Invocation{64, ops::LinearPolicy::A16Only},
        Invocation{65, ops::LinearPolicy::A16Only},
        Invocation{128, ops::LinearPolicy::A16Only},
        Invocation{129, ops::LinearPolicy::A16Only},
        Invocation{1024, ops::LinearPolicy::A16Only},
        Invocation{first_a4, ops::LinearPolicy::AllowA4},
        Invocation{17, ops::LinearPolicy::AllowA4},
        Invocation{8, ops::LinearPolicy::AllowA4},
        Invocation{9, ops::LinearPolicy::AllowA4},
        Invocation{16, ops::LinearPolicy::AllowA4},
        Invocation{32, ops::LinearPolicy::AllowA4},
        Invocation{64, ops::LinearPolicy::AllowA4},
        Invocation{96, ops::LinearPolicy::AllowA4},
        Invocation{128, ops::LinearPolicy::AllowA4},
        Invocation{129, ops::LinearPolicy::AllowA4},
        // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
        // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
        // single real token, which is the emptiest grid this route ever runs.
        Invocation{1023, ops::LinearPolicy::AllowA4},
        Invocation{1024, ops::LinearPolicy::AllowA4},
        Invocation{1025, ops::LinearPolicy::AllowA4},
    };
    return run_cases("NVFP4 linear_add", WeightFormat::NVFP4,
                     InvocationCase{
                         .n               = profile.n,
                         .k               = profile.k,
                         .seed            = profile.seed,
                         .invocations     = invocations,
                         .fixture         = FixturePattern::Hashed,
                         .rows            = Samples::BoundarySeven,
                         .columns         = Samples::BoundarySeven,
                         .scan_all_finite = false,
                     });
}

} // namespace

int main() {
    return ninfer::test::linear_common::run_test_main("NVFP4 linear_add", [] {
        return run_profile(Profile{.n = 5120, .k = 6144, .first_a4 = 17, .seed = 811U}) +
               run_profile(Profile{.n = 5120, .k = 17408, .first_a4 = 8, .seed = 821U});
    });
}
