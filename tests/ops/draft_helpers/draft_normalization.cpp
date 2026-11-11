#include "ops/suite.h"
void add_gathered_features_cases(ninfer::test::OpSuite&);
void add_rmsnorm_pack_tail_cases(ninfer::test::OpSuite&);
int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_gathered_features_cases(suite);
    add_rmsnorm_pack_tail_cases(suite);
    return suite.main(argc, argv);
}
