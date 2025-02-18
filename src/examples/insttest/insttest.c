#include <stdio.h>
#include <stdint.h>

void check(int pass) { printf(pass ? "Passed\n" : "Failed\n"); }

void
ldstubTest()
{
    printf("LDSTUB:\t\t");

    uint8_t volatile mem = 0x11;
    uint8_t rd = 0;
    uint8_t mem_o = mem, rd_o = rd;

    // Load one byte of memory into rd, zero extended, and set that byte in
    // memory to all ones.
    asm volatile ("ldstub [%[address]], %[rd]"
            : [rd] "=r" (rd)
            : [address] "r" (&mem));

    check(mem == 0xff && rd == mem_o);
}

void
swapTest()
{
    printf("SWAP:\t\t");

    uint32_t volatile mem = 0x01234567;
    uint32_t rd = 0x89ABCDEF;
    uint32_t mem_o = mem, rd_o = rd;

    // Swap the lower 32 bits of rd with a word of memory. Zero the upper
    // 32 bits of rd.
    asm volatile ("swap [%[address]], %[rd]"
            : [rd] "+r" (rd)
            : [address] "r" (&mem));

    check(mem == rd_o && rd == mem_o);
}

void
casFailTest()
{
    printf("CAS FAIL:\t");

    uint32_t rd = 0x00112233;
    uint32_t volatile mem = 0xffeeddcc;
    uint32_t rs2 = 0;
    uint32_t rd_o = rd, mem_o = mem, rs2_o = rs2;

    // Compare the lower 32 bits of rs2 with a word of memory. Since they are
    // different, leave memory unchanged and put the contents of memory in
    // rd, zero extended.
    asm volatile ("cas [%[address]], %[rs2], %[rd]"
            : [rd] "+r" (rd)
            : [address] "r" (&mem), [rs2] "r" (rs2));

    check(mem == mem_o && rd == mem_o);
}

void
casWorkTest()
{
    printf("CAS WORK:\t");

    uint32_t rd = 0x00112233;
    uint32_t volatile mem = 0xffeeddcc;
    uint32_t rs2 = 0xffeeddcc;
    uint32_t rd_o = rd, mem_o = mem, rs2_o = rs2;

    // Compare the lower 32 bits of rs2 with a word of memory. Since they are
    // equal, swap the value in lower 32 bits of rd with that word of memory.
    // Zero the upper 32 bits of rd.
    asm volatile ("cas [%[address]], %[rs2], %[rd]"
            : [rd] "+r" (rd)
            : [address] "r" (&mem), [rs2] "r" (rs2));

    check(mem == rd_o && rd == mem_o);
}

void
casxFailTest()
{
    printf("CASX FAIL:\t");

    uint64_t rd = 0x0011223344556677;
    uint64_t volatile mem = 0xffeeddccbbaa9988;
    uint64_t rs2 = 0;
    uint64_t rd_o = rd, mem_o = mem, rs2_o = rs2;

    // Compare rs2 with a doubleword of memory. Since they are different, leave
    // memory unchanged and put the contets of memory in rd.
    asm volatile ("casx [%[address]], %[rs2], %[rd]"
            : [rd] "+r" (rd)
            : [address] "r" (&mem), [rs2] "r" (rs2));

    check(mem == mem_o && rd == mem_o);
}

void
casxWorkTest()
{
    printf("CASX WORK:\t");

    uint64_t rd = 0x0011223344556677;
    uint64_t volatile mem = 0xffeeddccbbaa9988;
    uint64_t rs2 = 0xffeeddccbbaa9988;
    uint64_t rd_o = rd, mem_o = mem, rs2_o = rs2;

    // Compare rs2 with a doubleword of memory. Since they are equal, swap rd
    // with that doubleword of memory.
    asm volatile ("casx [%[address]], %[rs2], %[rd]"
            : [rd] "+r" (rd)
            : [address] "r" (&mem), [rs2] "r" (rs2));

    check(mem == rd_o && rd == mem_o);
}

void
ldtxTest()
{
    printf("LDTX:\t\t");

    uint64_t volatile mem[2] __attribute__((aligned(16))) = {
        0xffeeddccbbaa9988, 0x0011223344556677 };
    uint64_t rd1 = 0, rd2 = 0;
    uint64_t mem_o[2] = { mem[0], mem[1] };

    // LDDA will normally load a doubleword of memory into a register pair,
    // but when told to use ASI e2 (ASI_TWINX_PRIMARY) it will load two
    // doublewords which it will put into the register pair.
    asm volatile (
            "ldda [%[address]]0xe2, %%g2\n"
            "mov %%g2, %[rd1]\n"
            "mov %%g3, %[rd2]\n"
            : [rd1] "=r" (rd1), [rd2] "=r" (rd2)
            : [address] "r" (&mem[0])
            : "%g2", "%g3");

    check(rd1 == mem_o[0] && rd2 == mem_o[1]);
}

void
ldtwTest()
{
    printf("LDTW:\t\t");

    uint32_t volatile mem[2] __attribute__((aligned(8))) = {
        0x89abcdef, 0x01234567 };
    uint32_t rd1 = 0, rd2 = 0;
    uint64_t mem_o[2] = { mem[0], mem[1] };

    // Load two adjacent words from memory into the lower 32 bits of a pair
    // of registers, and zero their upper 32 bits.
    asm volatile (
            "ldd [%[address]], %%g2\n"
            "mov %%g2, %[rd1]\n"
            "mov %%g3, %[rd2]\n"
            : [rd1] "=r" (rd1), [rd2] "=r" (rd2)
            : [address] "r" (&mem[0])
            : "%g2", "%g3");
    check(rd1 == mem_o[0] && rd2 == mem_o[1]);
}

void
sttwTest()
{
    printf("STTW:\t\t");

    uint32_t volatile mem[2] __attribute__((aligned(8))) = { 0, 0 };
    uint64_t rd1 = 0x89abcdef, rd2 = 0x1234567;
    uint64_t rd1_o = rd1, rd2_o = rd2;

    // Store the lower 32 bits of a pair of registers into a single doubleword
    // of memory.
    asm volatile (
            "mov %[rd1], %%g2\n"
            "mov %[rd2], %%g3\n"
            "std %%g2, [%[address]]"
            :
            : [rd1] "r" (rd1), [rd2] "r" (rd2), [address] "r" (&mem[0])
            : "%g2", "%g3");

    check(mem[0] == rd1_o && mem[1] == rd2_o);
}

int
main()
{
    printf("Begining test of difficult SPARC instructions...\n");

    ldstubTest();
    swapTest();
    casFailTest();
    casWorkTest();
    casxFailTest();
    casxWorkTest();
    ldtxTest();
    ldtwTest();
    sttwTest();

    printf("Done\n");
    return 0;
}
