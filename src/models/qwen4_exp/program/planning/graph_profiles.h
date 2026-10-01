#pragma once
#include "models/qwen4_exp/program/program.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen4_exp::detail {

// Ordinary decode profiles over the execution frontier (the position of the decoded token).
// Each profile bounds the QSA indexer's complete-block envelope floor((frontier + 1) / block);
// envelopes within the indexer's full selection budget use its identity route (topology class
// 0), larger ones its scoring route (class 1).
[[nodiscard]] std::vector<GraphExecutionProfile>
ordinary_graph_profiles(std::uint32_t capacity, std::uint32_t indexer_block,
                        std::uint32_t indexer_selected_blocks);

// Complete-block envelope of a profile whose frontiers end at `max_frontier`.
[[nodiscard]] std::int32_t indexer_envelope_blocks(std::uint32_t max_frontier,
                                                   std::uint32_t indexer_block);

} // namespace ninfer::models::qwen4_exp::detail
