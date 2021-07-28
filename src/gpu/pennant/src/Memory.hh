/*
 * Memory.hh
 *
 *  Created on: Jul 3, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MEMORY_HH_
#define MEMORY_HH_

#include <cstdlib>


// Namespace Memory provides functions to allocate and free memory.
// Currently these are just wrappers around std::malloc and free,
// but they are abstracted here to make it easier to replace them
// if needed.

namespace Memory {

template<typename T>
inline T* alloc(const int count) {
    return (T*) std::malloc(count * sizeof(T));
}

template<typename T>
inline void free(T* ptr) {
    std::free(ptr);
}

};  // namespace Memory

#endif /* MEMORY_HH_ */
