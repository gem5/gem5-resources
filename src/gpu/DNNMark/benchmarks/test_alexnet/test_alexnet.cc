#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark(21, FLAGS_mmap);
  dnnmark.ParseAllConfig(FLAGS_config);

  dnnmark.Initialize();
  // Warm up
  if (FLAGS_warmup) {
    for (int i = 0; i < 5; i++) {
      dnnmark.Forward();
      dnnmark.Backward();
    }
  }
  dnnmark.GetTimer()->Clear();

  // Real benchmark
  dnnmark.Forward();
  dnnmark.Backward();
  dnnmark.GetTimer()->SumRecords();

  dnnmark.TearDown();

  LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
