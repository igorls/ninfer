#pragma once
#include "models/qwen4_exp/frontend.h"
#include "models/qwen4_exp/program/program.h"

namespace ninfer::models::qwen4_exp {

// The common request controller is instantiated once for this concrete model implementation.
struct RuntimeTypes {
    using Frontend                   = qwen4_exp::Frontend;
    using PreparedPrompt             = qwen4_exp::PreparedPrompt;
    using OutputSession              = qwen4_exp::OutputSession;
    using PublishedOutput            = qwen4_exp::PublishedOutput;
    using SequencePlanner            = qwen4_exp::SequencePlanner;
    using SequencePlan               = qwen4_exp::SequencePlan;
    using RequestBasePlan            = qwen4_exp::RequestBasePlan;
    using AdmissionCandidate         = qwen4_exp::AdmissionCandidate;
    using ResourcePlan               = qwen4_exp::ResourcePlan;
    using PersistentBackfillProof    = qwen4_exp::PersistentBackfillProof;
    using SequenceHandle             = qwen4_exp::SequenceHandle;
    using ContinuationHandle         = qwen4_exp::ContinuationHandle;
    using SharedPrefixHandle         = qwen4_exp::SharedPrefixHandle;
    using CaptureOffer               = qwen4_exp::CaptureOffer;
    using CacheSessionKey            = qwen4_exp::PreparedSessionKey;
    using ContinuationSummary        = qwen4_exp::ContinuationSummary;
    using SharedPrefixSummary        = qwen4_exp::SharedPrefixSummary;
    using PressurePlanningSession    = qwen4_exp::PressurePlanningSession;
    using PressureTargetHandle       = qwen4_exp::PressureTargetHandle;
    using AssessedPressureTarget     = qwen4_exp::AssessedPressureTarget;
    using CapturePressurePlan        = qwen4_exp::CapturePressurePlan;
    using MaterializationResult      = qwen4_exp::MaterializationResult;
    using ContextTransactionProgress = qwen4_exp::ContextTransactionProgress;
    using CaptureAssessment          = qwen4_exp::CaptureAssessment;
    using ActiveCaptureResult        = qwen4_exp::ActiveCaptureResult;
    using PendingBatch               = qwen4_exp::PendingBatch;
    using StartResult                = qwen4_exp::StartResult;
    using PrefillProgress            = qwen4_exp::PrefillProgress;
    using CommitResult               = qwen4_exp::CommitResult;
    using DiscardResult              = qwen4_exp::DiscardResult;
    using FinishResult               = qwen4_exp::FinishResult;
    using AbortResult                = qwen4_exp::AbortResult;
    using ReleaseResult              = qwen4_exp::ReleaseResult;
    using Program                    = qwen4_exp::Program;
};

} // namespace ninfer::models::qwen4_exp
