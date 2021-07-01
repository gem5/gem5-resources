#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark(3, FLAGS_mmap);
  dnnmark.ParseAllConfig(FLAGS_config);
  dnnmark.Initialize();
  dnnmark.Forward();
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
