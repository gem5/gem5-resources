#ifndef _COMPLIANCE_MODEL_H
#define _COMPLIANCE_MODEL_H

#if XLEN == 64
  #define ALIGNMENT 3
#else
  #define ALIGNMENT 2
#endif

#define RVMODEL_DATA_SECTION \
tohost_file:                                                            \
        .asciz "signature";                                             \
tohost_data:                                                            \
        .byte 0;                                                        \
        .pushsection .tohost,"aw",@progbits;                            \
        .align 8; .global tohost; tohost: .dword 0;                     \
        .align 8; .global fromhost; fromhost: .dword 0;                 \
        .popsection;                                                    \
        .align 8; .global begin_regstate; begin_regstate:               \
        .word 128;                                                      \
        .align 8; .global end_regstate; end_regstate:                   \
        .word 4;

//RV_COMPLIANCE_HALT
#define RVMODEL_HALT                                              \
  addi x1, x1, 4;                                                 \
  li x1, 0;                                                       \
  write_tohost:                                                   \
    /* Write signature memory segment to host file */             \
    sw x1, tohost, t1;                                            \
    add a2, zero, zero;                                           \
    la t1, rvtest_sig_begin;                                      \
    la t0, rvtest_sig_end;                                        \
  write:                                                          \
    /* Write 16 bytes signature data to file */                   \
    bge t1, t0, shutdown;                                         \
    add a0, t1, zero;                                             \
    li a1, 16;                                                    \
    la a3, tohost_file;                                           \
    /* M5Op of write file */                                      \
    .long 0x9E00007B;                                             \
    addi t1, t1, 16;                                              \
    addi a2, a2, 16;                                              \
    j write;                                                      \
  shutdown:                                                       \
    /* shutdown gem5 */                                           \
    li a0, 0;                                                     \
    /* M5Op of exit simulation */                                 \
    .long 0x4200007B;                                             \
  self_loop:  j self_loop;                                        \

#define RVMODEL_BOOT

//RV_COMPLIANCE_DATA_BEGIN
#define RVMODEL_DATA_BEGIN                                              \
  .align 4; .global begin_signature; begin_signature:

//RV_COMPLIANCE_DATA_END
#define RVMODEL_DATA_END                                                      \
  .align 4; .global end_signature; end_signature:  \
  RVMODEL_DATA_SECTION                                                        \

//RVTEST_IO_INIT
#define RVMODEL_IO_INIT
//RVTEST_IO_WRITE_STR
#define RVMODEL_IO_WRITE_STR(_R, _STR)
//RVTEST_IO_CHECK
#define RVMODEL_IO_CHECK()
//RVTEST_IO_ASSERT_GPR_EQ
#define RVMODEL_IO_ASSERT_GPR_EQ(_S, _R, _I)
//RVTEST_IO_ASSERT_SFPR_EQ
#define RVMODEL_IO_ASSERT_SFPR_EQ(_F, _R, _I)
//RVTEST_IO_ASSERT_DFPR_EQ
#define RVMODEL_IO_ASSERT_DFPR_EQ(_D, _R, _I)

#define RVMODEL_SET_MSW_INT       \
 li t1, 1;                         \
 li t2, 0x2000000;                 \
 sw t1, 0(t2);

#define RVMODEL_CLEAR_MSW_INT     \
 li t2, 0x2000000;                 \
 sw x0, 0(t2);

#define RVMODEL_CLEAR_MTIMER_INT

#define RVMODEL_CLEAR_MEXT_INT

#endif // _COMPLIANCE_MODEL_H

