# Changelog

## Version 1.0.0

* Initial release

## Known Issues

* For the instruction `csrr a5,mcycle`, on running in gem5, the error message: `mcycle is not accessible in 0` was received. This was fixed by commenting the function calls `Start_Timer();` and `Stop_Timer();` in `dhrystone_main.c`.