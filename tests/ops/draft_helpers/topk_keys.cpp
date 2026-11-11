#include "ops/linear/support.h"
void add_topk_keys_cases(ninfer::test::linear_suite::Suite&);
int main(int argc, char** argv) {
    ninfer::test::linear_suite::Suite suite;
    add_topk_keys_cases(suite);
    return suite.main(argc, argv);
}
