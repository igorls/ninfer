#pragma once

// Qwen4Exp prompts and outputs are the Qwen3.5 Frontend's: same tokenizer family, chat-template
// mechanics, MRoPE prompt layout, Vision preprocessing and output channels. The package names
// them in its own namespace so its Program reads like any other package's.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/vision_control.h"

namespace ninfer::models::qwen4_exp {

namespace frontend = ::ninfer::models::qwen3_5::frontend;

using qwen3_5::Frontend;
using qwen3_5::OutputSession;
using qwen3_5::PreparedCacheOpportunity;
using qwen3_5::PreparedContextCache;
using qwen3_5::PreparedMediaPayload;
using qwen3_5::PreparedPrompt;
using qwen3_5::PreparedPromptAccess;
using qwen3_5::PreparedPromptData;
using qwen3_5::PreparedSessionKey;
using qwen3_5::PromptIdentity;
using qwen3_5::PromptModality;
using qwen3_5::PublishedOutput;
using qwen3_5::RewriteCheckpointKind;
using qwen3_5::RewriteCheckpointSpec;
using qwen3_5::TokenSpan;
using qwen3_5::VisionControl;
using qwen3_5::VisionControlPlan;
using qwen3_5::VisionGrid;
using qwen3_5::VisionItem;
using qwen3_5::VisionItemControl;
using qwen3_5::VisionItemControlPlan;
using qwen3_5::build_vision_control;
using qwen3_5::kMaximumPromptVisionTokens;
using qwen3_5::kMaximumVisionItemTokens;
using qwen3_5::kPreparedVisionPatchFeatures;
using qwen3_5::make_frontend;
using qwen3_5::plan_vision_control;

} // namespace ninfer::models::qwen4_exp
