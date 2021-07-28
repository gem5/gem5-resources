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

#include "HSA_Interface.h"

void print_agent_dispatch_packet(hsa_agent_dispatch_packet_t* pkt)
{

    printf("Packet \t\t%p\n",
        (void *)pkt);
    printf("Packet 16t\t\t%p\n",
        (uint16_t *)pkt);
    printf("Packet 32t\t\t%p\n",
        (uint32_t *)pkt);
    printf("Packet void**\t\t%p\n",
        (void **)pkt);
    printf("%p header: \t\t%hu\n",
        (void *)(&(pkt->header )),pkt->header );
    printf("%p type: \t\t%hu\n",
        (void *)(&(pkt->type )),pkt->type );
    printf("%p reserved0: \t\t%u\n",
        (void *)(&(pkt->reserved0 )),pkt->reserved0 );
    printf("%p return_address: \t\t%p\n",
        (void *)(&(pkt->return_address )),pkt->return_address );
    printf("%p arg[0]: \t\t%lu\n",
        (void *)(&(pkt->arg[0] )),pkt->arg[0] );
    printf("%p arg[1]: \t\t%lu\n",
        (void *)(&(pkt->arg[1] )),pkt->arg[1] );
    printf("%p arg[2]: \t\t%lu\n",
        (void *)(&(pkt->arg[2] )),pkt->arg[0] );
    printf("%p arg[3]: \t\t%lu\n",
        (void *)(&(pkt->arg[3] )),pkt->arg[1] );
    printf("%p reserved2: \t\t%lu\n",
        (void *)(&(pkt->reserved2 )),pkt->reserved2 );
    printf("%p completion_signal: \t\t%lu\n",
        (void *)(&(pkt->completion_signal )),pkt->completion_signal.handle );


    fflush(stdout);
}

void agent_disp_packet_store_release(uint16_t* packet, uint16_t header) {
    __atomic_store_n(packet, header, __ATOMIC_RELEASE);
}

uint16_t header(hsa_packet_type_t type) {
    uint16_t header = type << HSA_PACKET_HEADER_TYPE;
    header |=
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |=
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    return header;
}

hsa_status_t get_kernel_agent(hsa_agent_t agent, void* data) {
    uint32_t features = 0;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features);
    if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
        // Store kernel agent in the application-provided buffer and return
        hsa_agent_t* ret = (hsa_agent_t*) data;
        *ret = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    // Keep iterating
    return HSA_STATUS_SUCCESS;
}

void signal_wait(hsa_signal_t signal)
{
    while (hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0,
        UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0);
    // while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
    //  UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0);
}

void initialize_agent_dispatch_packet(
    hsa_agent_dispatch_packet_t* packet,
    size_t header_size
    )
{
    // Reserved fields, private and group memory,
    // and completion signal are all set to 0.
    memset(((uint8_t*) packet) + header_size, 0,
        sizeof(hsa_agent_dispatch_packet_t) - header_size);
}

HSA_Interface::HSA_Interface(){

    printf("INFO:: Setting up HSA Interface:\n");

    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name); fflush(stdout);
    #ifdef __HIP_PLATFORM_HCC__
      printf ("info: architecture on AMD GPU device is: %d\n",
        props.gcnArch); fflush(stdout);
    #endif

    printf ("INFO:: hsa_iterate_agents\n"); fflush(stdout);
    hsa_agent_t kernel_agent;
    hsa_iterate_agents(get_kernel_agent, &kernel_agent);
    printf ("INFO:: hsa_queue_create\n"); fflush(stdout);
    hsa_queue_create(kernel_agent, 4, HSA_QUEUE_TYPE_SINGLE,
        NULL, NULL, 0, 0, &queue);
    printf ("INFO:: hsa_queue_add_write_index_relaxed\n"); fflush(stdout);
    hsa_queue_add_write_index_relaxed(queue, 1);

    packet_id = 0;

    printf("INFO:: Creating Stream\n");fflush(stdout);
    stream = 0;
    hipStreamCreate(&stream);
}

void HSA_Interface::steal_kernel_signal(uint32_t kid)
{
    hsa_agent_dispatch_packet_t * packet =
        (hsa_agent_dispatch_packet_t*) queue->base_address + packet_id;
    // Populate fields in kernel dispatch packet, except for the header,
    // the setup, and the completion signal fields
    initialize_agent_dispatch_packet(packet,sizeof(uint16_t));

    uint64_t kernel_completion_signal_addr;
    packet->type = AGENT_DISPATCH_PACKET_STEAL_KERNEL_SIGNAL;
    packet->return_address = &kernel_completion_signal_addr;
    packet->arg[0] = kid; //This field is for the kernel id.

    //Create thief packet wait signal
    hsa_signal_create(1, 0, NULL, &packet->completion_signal);

    agent_disp_packet_store_release((uint16_t*) packet,
        header(HSA_PACKET_TYPE_AGENT_DISPATCH));

    print_agent_dispatch_packet(packet);

    //Send thief packet
    hsa_signal_store_screlease(queue->doorbell_signal, packet_id);

    signal_wait(packet->completion_signal);
    printf("INFO:: Done Waiting on Thief Signal\n");

    hsa_signal_t * new_signal = new hsa_signal_t;
    new_signal->handle = kernel_completion_signal_addr;
    m_kernel_signals.push_back(new_signal);

    packet_id++;
}

void HSA_Interface::wait_kernel(uint32_t kid)
{
    signal_wait(*(m_kernel_signals[kid]));
}