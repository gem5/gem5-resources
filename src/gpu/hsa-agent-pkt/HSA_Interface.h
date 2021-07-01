/*
Copyright (c) 2020 University of Maryland
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met: redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer;
redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution;
neither the name of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "hip/hip_runtime.h"

#include <include/hsa/hsa.h>

#include <vector>

#define AGENT_DISPATCH_PACKET_NOP 0
#define AGENT_DISPATCH_PACKET_STEAL_KERNEL_SIGNAL 1

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) {\
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n",\
      hipGetErrorString(error), error,__FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
    }\
}

void print_agent_dispatch_packet(hsa_agent_dispatch_packet_t* pkt);
void agent_disp_packet_store_release(uint16_t* packet, uint16_t header);
uint16_t header(hsa_packet_type_t type);
hsa_status_t get_kernel_agent(hsa_agent_t agent, void* data);
void signal_wait( hsa_signal_t signal);
void initialize_agent_dispatch_packet(
    hsa_agent_dispatch_packet_t* packet,
    size_t header_size
    );

//Class for interacting with kernel agent and creating pipes
class HSA_Interface {

public:
    HSA_Interface();
    ~HSA_Interface(){};

    void steal_kernel_signal(uint32_t kid);
    void wait_kernel(uint32_t kid);

    hipStream_t getStream() {return stream;}

private:
    hsa_queue_t * queue;
    hipStream_t stream;
    hipDeviceProp_t props;

    //Store Kernel Signals for multuple launches
    std::vector<hsa_signal_t *> m_kernel_signals;

    //Each packet created will have an ID associated with it.
    //It is used to index into the hsa queue.
    uint64_t packet_id;

};

