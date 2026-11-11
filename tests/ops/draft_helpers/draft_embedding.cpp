#include "ops/suite.h"
void add_embedding_cases(ninfer::test::OpSuite&);
int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_embedding_cases(suite);
    return suite.main(argc, argv);
}
