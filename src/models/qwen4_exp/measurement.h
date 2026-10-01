#pragma once

#include "models/qwen4_exp/model.h"

#include <string>

namespace ninfer::models::qwen4_exp {

// Measurement applicability, independent of checkpoint names, object IDs, payload values and
// device addresses. This key never selects or admits an execution implementation.
[[nodiscard]] std::string prefill_signature(const Model& model);

} // namespace ninfer::models::qwen4_exp
