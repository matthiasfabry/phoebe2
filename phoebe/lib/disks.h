#pragma once



#include <iostream>
#include <vector>


#include "triang_mesh.h"
#include "utils.h"
#include "cvec.h"


template <class T>
int build_disk_mesh(
    T inner_r,
    T outer_r,
    T height,
    const unsigned &max_triangles,
    std::vector<T3Dpoint<T>> &V,
    std::vector<T3Dpoint<T>> &NatV,
    std::vector<T3Dpoint<int>> &Tr,
    std::vector<T> *GatV = 0,
    const T &init_phi = 0)
{
    T3Dpoint<T> first = {inner_r, 0, -height / 2};
    T3Dpoint<T> second = {outer_r, 0, -height / 2};
    T3Dpoint<T> third = {outer_r, 0, height / 2};
    T3Dpoint<T> fourth = {inner_r, 0, height / 2};

    int segments = max_triangles / 8; // Each segment will have 8 triangles (4 for the top and 4 for the bottom)
    std::cout << "sections = " << segments << std::endl;
    double dphi = 8 * 2 * M_PI / max_triangles;
    std::cout << "dphi = " << dphi << std::endl;

    V.emplace_back(first);
    V.emplace_back(second);
    V.emplace_back(third);
    V.emplace_back(fourth);

    for (int i = 0; i < segments; i++) {
        T phi = i * dphi;
        T cos_phi = std::cos(phi);
        T sin_phi = std::sin(phi);

        T3Dpoint<T> p1{inner_r * cos_phi, inner_r * sin_phi, -height / 2};
        T3Dpoint<T> p2{outer_r * cos_phi, outer_r * sin_phi, -height / 2};
        T3Dpoint<T> p3{outer_r * cos_phi, outer_r * sin_phi, height / 2};
        T3Dpoint<T> p4{inner_r * cos_phi, inner_r * sin_phi, height / 2};

        V.emplace_back(p1);
        V.emplace_back(p2);
        V.emplace_back(p3);
        V.emplace_back(p4);
    }

    // build the triangles
    for (int i = 0; i < segments; i++) {
        int idx = i * 4;
        Tr.emplace_back(idx, idx + 1, idx + 5);
        Tr.emplace_back(idx, idx + 5, idx + 4);

        Tr.emplace_back(idx, idx + 7, idx + 3);
        Tr.emplace_back(idx, idx + 4, idx + 7);

        Tr.emplace_back(idx + 2, idx + 5, idx + 1);
        Tr.emplace_back(idx + 2, idx + 6, idx + 5);

        Tr.emplace_back(idx + 2, idx + 7, idx + 6);
        Tr.emplace_back(idx + 2, idx + 3, idx + 7);
    }

    // // last triangles wrap indices
    int idx = segments * 4;
    Tr.emplace_back(idx, idx + 1, 1);
    Tr.emplace_back(idx, 2, 0);

    Tr.emplace_back(idx, 3, idx + 3);
    Tr.emplace_back(idx, 0, 3);
    
    Tr.emplace_back(idx + 2, 1, idx + 1);
    Tr.emplace_back(idx + 2, 2, 1);

    Tr.emplace_back(idx + 2, 3, 2);
    Tr.emplace_back(idx + 2, idx + 3, 3);


    return 0;
}