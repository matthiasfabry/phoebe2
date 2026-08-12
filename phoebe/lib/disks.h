#pragma once
/*
  Disk tessellation with roughly-equilateral triangles, with
  per-vertex normals (NatV) so that area/volume can be computed with
  mesh_area_volume() from triang_mesh.h.

  mesh_area_volume() uses the supplied vertex normal as the orientation
  reference for each triangle (flipping the triangle's contribution if
  its winding disagrees with that normal), so it self-corrects for any
  winding mistakes in Tr -- PROVIDED NatV is actually correct. That's
  the reason the four surfaces (top, bottom, inner wall, outer wall) are
  built as fully independent vertex sets below, even though they are
  numerically coincident at the seams: a seam is a sharp edge, so a
  single shared vertex there has no well-defined single normal, and using
  the wrong one as an orientation reference for a triangle could flip its
  sign incorrectly. Independent per-surface vertices sidestep that.
*/

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "triang_mesh.h" // T3Dpoint<T>, mesh_area_volume

template <class T>
struct Tdisk_mesh_equilateral
{

    T inner_r, outer_r, height;

    Tdisk_mesh_equilateral(const T &inner_r_, const T &outer_r_, const T &height_)
        : inner_r(inner_r_), outer_r(outer_r_), height(height_) {}

    // ring of n points at radius r, height z, all sharing the constant normal (0,0,nz)
    static int add_ring_flat(std::vector<T3Dpoint<T>> &V, std::vector<T3Dpoint<T>> &NatV,
                             T r, T z, int n, T nz)
    {
        int start = (int)V.size();
        for (int i = 0; i < n; ++i)
        {
            T phi = T(2) * T(M_PI) * T(i) / n;
            V.emplace_back(r * std::cos(phi), r * std::sin(phi), z);
            NatV.emplace_back(T(0), T(0), nz);
        }
        return start;
    }

    // ring of n points at radius r, height z, radial normal (outward if 'outward')
    static int add_ring_wall(std::vector<T3Dpoint<T>> &V, std::vector<T3Dpoint<T>> &NatV,
                             T r, T z, int n, bool outward)
    {
        int start = (int)V.size();
        T sgn = outward ? T(1) : T(-1);
        for (int i = 0; i < n; ++i)
        {
            T phi = T(2) * T(M_PI) * T(i) / n, c = std::cos(phi), s = std::sin(phi);
            V.emplace_back(r * c, r * s, z);
            NatV.emplace_back(sgn * c, sgn * s, T(0));
        }
        return start;
    }

    /*
      Zipper: stitch two coaxial rings with possibly different point counts
      (nA, nB) into a triangle strip by walking both by normalized angle
      fraction. Produces nA+nB triangles.
    */
    static void bridge_rings(std::vector<T3Dpoint<int>> &Tr,
                             int startA, int nA, int startB, int nB, bool flip)
    {
        int i = 0, j = 0;
        for (int step = 0, total = nA + nB; step < total; ++step)
        {
            int a0 = startA + i % nA, a1 = startA + (i + 1) % nA;
            int b0 = startB + j % nB;

            double fa = double(i + 1) / nA, fb = double(j + 1) / nB;

            if (fa <= fb)
            {
                if (!flip)
                    Tr.emplace_back(a0, a1, b0);
                else
                    Tr.emplace_back(a0, b0, a1);
                ++i;
            }
            else
            {
                int b1 = startB + (j + 1) % nB;
                if (!flip)
                    Tr.emplace_back(a0, b1, b0);
                else
                    Tr.emplace_back(a0, b0, b1);
                ++j;
            }
        }
    }

    // vertical cylindrical strip at fixed radius r, n points around, N_h subdivisions,
    // fully independent vertices/normals (not shared with the top/bottom faces)
    void build_wall(std::vector<T3Dpoint<T>> &V, std::vector<T3Dpoint<T>> &NatV,
                    std::vector<T3Dpoint<int>> &Tr,
                    T r, int n, int N_h, bool outward) const
    {

        std::vector<int> ring(N_h + 1);
        for (int j = 0; j <= N_h; ++j)
            ring[j] = add_ring_wall(V, NatV, r, -height / 2 + height * T(j) / N_h, n, outward);

        for (int j = 0; j < N_h; ++j)
            for (int i = 0; i < n; ++i)
            {
                int i1 = (i + 1) % n;
                int a = ring[j] + i, b = ring[j] + i1;
                int c = ring[j + 1] + i, d = ring[j + 1] + i1;
                if (outward)
                {
                    Tr.emplace_back(a, b, d);
                    Tr.emplace_back(a, d, c);
                }
                else
                {
                    Tr.emplace_back(a, d, b);
                    Tr.emplace_back(a, c, d);
                }
            }
    }

    void make_mesh(int max_triangles,
                   std::vector<T3Dpoint<T>> &V,
                   std::vector<T3Dpoint<T>> &NatV,
                   std::vector<T3Dpoint<int>> &Tr) const
    {

        V.clear();
        NatV.clear();
        Tr.clear();

        // --- target edge length from desired triangle count & total area ---
        T annulus_area = T(M_PI) * (outer_r * outer_r - inner_r * inner_r); // one face
        T limb_area = T(2) * T(M_PI) * height * (inner_r + outer_r);        // both limbs
        T total_area = T(2) * annulus_area + limb_area;

        T s = std::sqrt(T(4) * total_area / (std::sqrt(T(3)) * max_triangles));

        // --- radial ring layout for top/bottom faces ---
        int N_r = std::max(1, (int)std::lround((outer_r - inner_r) / s));

        std::vector<T> radii(N_r + 1);
        std::vector<int> n_phi(N_r + 1);
        for (int k = 0; k <= N_r; ++k)
        {
            radii[k] = inner_r + (outer_r - inner_r) * T(k) / N_r;
            n_phi[k] = std::max(3, (int)std::lround(T(2) * T(M_PI) * radii[k] / s));
        }

        std::vector<int> top_start(N_r + 1), bot_start(N_r + 1);
        for (int k = 0; k <= N_r; ++k)
        {
            top_start[k] = add_ring_flat(V, NatV, radii[k], height / 2, n_phi[k], T(1));
            bot_start[k] = add_ring_flat(V, NatV, radii[k], -height / 2, n_phi[k], T(-1));
        }

        for (int k = 0; k < N_r; ++k)
            bridge_rings(Tr, top_start[k], n_phi[k], top_start[k + 1], n_phi[k + 1], false);

        for (int k = 0; k < N_r; ++k)
            bridge_rings(Tr, bot_start[k], n_phi[k], bot_start[k + 1], n_phi[k + 1], true);

        // --- limbs, vertically subdivided, independent vertices/normals ---
        int N_h = std::max(1, (int)std::lround(height / s));

        build_wall(V, NatV, Tr, inner_r, n_phi[0], N_h, false);
        build_wall(V, NatV, Tr, outer_r, n_phi[N_r], N_h, true);

        // std::cout << "target edge length s = " << s
        //           << ", radial rings = " << N_r
        //           << ", wall subdivisions = " << N_h
        //           << ", vertices = " << V.size()
        //           << ", triangles = " << Tr.size() << std::endl;
    }

    /*
      Area and volume of the closed disk mesh, computed via mesh_area_volume
      (triang_mesh.h), which uses NatV as the orientation reference per
      triangle -- so it self-corrects for any winding inconsistency, as long
      as NatV is correct (which it is here, by construction).

      Also prints the analytic expected values for a flat annulus x height
      as a sanity check:
        area   = 2*pi*(outer_r^2 - inner_r^2) + 2*pi*height*(inner_r+outer_r)
        volume = pi*(outer_r^2 - inner_r^2)*height
    */
    void compute_area_volume(std::vector<T3Dpoint<T>> &V,
                             std::vector<T3Dpoint<T>> &NatV,
                             std::vector<T3Dpoint<int>> &Tr,
                             T &area, T &volume) const
    {
        T av[2];
        mesh_area_volume(V, NatV, Tr, av);
        area = av[0];
        volume = av[1];

        T area_analytic = T(2) * T(M_PI) * (outer_r * outer_r - inner_r * inner_r) + T(2) * T(M_PI) * height * (inner_r + outer_r);
        T volume_analytic = T(M_PI) * (outer_r * outer_r - inner_r * inner_r) * height;

        std::cout << "area:   mesh = " << area << ", analytic = " << area_analytic
                  << " (rel. err " << std::abs(area / area_analytic - 1) << ")\n"
                  << "volume: mesh = " << volume << ", analytic = " << volume_analytic
                  << " (rel. err " << std::abs(volume / volume_analytic - 1) << ")\n";
    }
    /*
    Per-triangle areas, one entry per row of Tr, in the same order as Tr.

    Uses the NatV-free overload of mesh_attributes(): the area of a
    triangle is |cross product|/2 regardless of winding order, so unlike
    volume it needs no orientation reference at all -- this is robust
    even if some triangles in Tr end up wound the "wrong" way.
    */
    void compute_triangle_areas(std::vector<T3Dpoint<T>> &V,
                                std::vector<T3Dpoint<int>> &Tr,
                                std::vector<T> &areas) const
    {
        mesh_attributes(V, Tr, &areas);
    }
};