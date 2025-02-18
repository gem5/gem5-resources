# Changelog

## Version 1.0.0

* Initial release

## Known Issues

* This benchmark uses `read_csr` and `write_csr` functions. These use inline assembly and thus, do not translate to gem5. Currently, a gem5 equivalent is being investigated.