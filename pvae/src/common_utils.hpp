#pragma once

#include <mutex>

std::mutex rand_mtx;
int randint(int min, int max){
    std::lock_guard<std::mutex> lock(rand_mtx);
    return (min + (std::rand() % (max - min)));
}
