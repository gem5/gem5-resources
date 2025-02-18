ENTRY(_start)
SECTIONS
{
  .text : {
    */bootloader.o(.text)
    *(.text)
    *(.rodata)
    *(.data)
    *(COMMON)
  }
  .bss : { *(.bss) }
  heap_low = .;
  . = . + 0x1000000;
  heap_top = .;
  . = . + 0x1000000;
  stack_top = .;
}
