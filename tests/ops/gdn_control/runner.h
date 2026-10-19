#pragma once

#include "fixtures.h"

namespace ninfer::test::gdn_control {
int run(Profile profile, Case test_case, bool normalize_input, CudaExecutionView execution);
int verify_column_invariance(CudaExecutionView execution);
} // namespace ninfer::test::gdn_control
