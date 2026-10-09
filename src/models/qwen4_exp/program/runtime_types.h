#pragma once
#include "models/qwen4_exp/frontend.h"
#include "models/qwen4_exp/program/program.h"

namespace ninfer::models::qwen4_exp {
struct RuntimeTypes {
    using Frontend          = qwen4_exp::Frontend;
    using PreparedPrompt    = qwen4_exp::PreparedPrompt;
    using OutputSession     = qwen4_exp::OutputSession;
    using PublishedOutput   = qwen4_exp::PublishedOutput;
    using SequencePlanner   = qwen4_exp::SequencePlanner;
    using SequencePlan      = qwen4_exp::SequencePlan;
    using RequestBasePlan   = qwen4_exp::RequestBasePlan;
    using SequenceHandle    = qwen4_exp::SequenceHandle;
    using CheckpointHandle  = qwen4_exp::CheckpointHandle;
    using CheckpointSummary = qwen4_exp::CheckpointSummary;
    using SourceCandidate   = qwen4_exp::SourceCandidate;
    using ResumeState       = qwen4_exp::ResumeState;
    using ExecutionUnit     = qwen4_exp::ExecutionUnit;
    using ExecutionUnitKind = qwen4_exp::ExecutionUnitKind;
    using ContextProgress   = qwen4_exp::ContextProgress;
    using PendingBatch      = qwen4_exp::PendingBatch;
    using PrefillProgress   = qwen4_exp::PrefillProgress;
    using ReplayProgress    = qwen4_exp::ReplayProgress;
    using CommitResult      = qwen4_exp::CommitResult;
    using DiscardResult     = qwen4_exp::DiscardResult;
    using FinishResult      = qwen4_exp::FinishResult;
    using AbortResult       = qwen4_exp::AbortResult;
    using Program           = qwen4_exp::Program;
    using CacheSessionKey   = qwen4_exp::PreparedSessionKey;
};
} // namespace ninfer::models::qwen4_exp
