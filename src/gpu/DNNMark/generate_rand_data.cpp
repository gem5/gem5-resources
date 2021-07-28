#include <cstdlib>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <ios>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    uint64_t SIZE = (uint64_t)2*1024*1024*1024/sizeof(float); // 2GB
    std::vector<std::vector<float>> vecs(1);
    for(auto &vec : vecs)
        vec.reserve(SIZE);

    srand(1);
    for(int j = 0; j < vecs.size(); j++)
        for(int i = 0; i < SIZE; i++) {
            float r = rand();
            vecs[j].push_back(r);
        }

    for(int i = 0; i < vecs.size(); i++)
    {
        std::ofstream fout("mmap.bin", std::ios::out | std::ios::binary);
        fout.write((char *)&vecs[i][0], vecs[i].size() * sizeof(float));
        fout.close();
    }

    return 0;
}
