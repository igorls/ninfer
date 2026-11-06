#include "causal/fixture.h"
#include "kv_cache_storage.h"

int run_softmax_attention_causal_cache_tests(std::optional<ninfer::KvCacheStorage> selected) {
    ninfer::test::OpSuite suite;
    ninfer::test::causal_attention::add_cases(suite);
    char name[] = "causal_attention";
    char filter[] = "--format";
    std::string format = selected ? ninfer::test::kv_cache_storage_name(*selected) : "";
    char* arguments[]{name, filter, format.data()};
    return suite.main(selected ? 3 : 1, arguments);
}
