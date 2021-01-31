#include <iostream>
#include "common.h"
#include "dnnmark.h"
#include "usage.h"

using namespace dnnmark;

int main(int argc, char **argv) {
  //google::ParseCommandLineFlags(&argc, &argv, true);
  //google::InitGoogleLogging(argv[0]);
  //FLAGS_logtostderr = FLAGS_debuginfo;
  INIT_FLAGS(argc, argv);
  INIT_LOG(argv);
  LOG(INFO) << "DNNMark suites: Start...";
  DNNMark<TestType> dnnmark(FLAGS_mmap);
  dnnmark.ParseGeneralConfig(FLAGS_config);
  dnnmark.ParseLayerConfig(FLAGS_config);
  dnnmark.Initialize();
  dnnmark.Backward();
  dnnmark.GetTimer()->SumRecords();
  dnnmark.TearDown();
  LOG(INFO) << "Total running time(ms): " << dnnmark.GetTimer()->GetTotalTime();
  LOG(INFO) << "DNNMark suites: Tear down...";
  return 0;
}
