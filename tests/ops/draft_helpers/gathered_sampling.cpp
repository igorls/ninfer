#include "ops/suite.h"
void add_gathered_selector_cases(ninfer::test::OpSuite&);
void add_vocabulary_sparse_cases(ninfer::test::OpSuite&);
int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_gathered_selector_cases(suite);
    add_vocabulary_sparse_cases(suite);
    return suite.main(argc, argv);
}
