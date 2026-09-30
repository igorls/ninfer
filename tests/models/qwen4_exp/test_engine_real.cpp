#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// The public Engine loads a Qwen4Exp artifact completely and then refuses to become an Engine,
// because the package has no execution Program yet.
int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    std::map<ninfer::StartupPhase, ninfer::StartupStatus> phases;
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.enable_vision             = true;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.startup_observer.callback = [&](const ninfer::StartupEvent& event) {
        if (event.status != ninfer::StartupStatus::Progress) { phases[event.phase] = event.status; }
    };
    try {
        ninfer::Engine engine(options);
        std::cerr << "a Qwen4Exp Engine was constructed without an execution Program\n";
        return 1;
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        std::cout << "Engine refused: " << message << '\n';
        const auto completed = [&](ninfer::StartupPhase phase) {
            const auto found = phases.find(phase);
            return found != phases.end() && found->second == ninfer::StartupStatus::Complete;
        };
        const auto failed = phases.find(ninfer::StartupPhase::TargetFinalize);
        if (message.find("Qwen4ExpForCausalLM") == std::string::npos ||
            message.find("no qwen4_exp execution Program") == std::string::npos ||
            !completed(ninfer::StartupPhase::TargetPlan) ||
            !completed(ninfer::StartupPhase::WeightsMaterialize) ||
            !completed(ninfer::StartupPhase::FrontendInitialize) || failed == phases.end() ||
            failed->second != ninfer::StartupStatus::Failed ||
            phases.contains(ninfer::StartupPhase::ProgramInitialize)) {
            std::cerr << "the Engine did not fail explicitly after a complete load\n";
            return 1;
        }
    }
    return 0;
}
