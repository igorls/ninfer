#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    const std::array<std::uint64_t, 3> shape = {2, 128, 64};
    const auto geometry =
        ninfer::artifact::block_scale_bank_geometry(ninfer::artifact::NumericFormat::NVFP4, shape);
    std::vector<std::byte> payload(geometry.encoded_bytes);
    const float divisors[2] = {2.0F, 3.0F};
    std::memcpy(payload.data() + geometry.weight_divisor_offset, divisors, sizeof(divisors));

    const Nvfp4ExpertBankView bank =
        make_nvfp4_expert_bank_view(payload.data(), payload.size(), 2, 128, 64);
    const Nvfp4ExpertMatrixView second = bank.expert(1);
    if (second.codes != payload.data() + 128 * 64 / 2 ||
        second.scales != payload.data() + geometry.scale_plane_offset + 128 * 64 / 16 ||
        *second.weight_scale_divisor != 3.0F || second.input_scale_divisor != 1.0F ||
        second.rows != 128 || second.columns != 64) {
        std::cerr << "NVFP4 expert bank view has the wrong plane geometry\n";
        return 1;
    }
    return 0;
}
