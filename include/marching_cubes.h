#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "SimpleMesh.h"
#include "Volume.h"

struct MC_Gridcell {
    Vector3d p[8];
    double val[8];
};

struct MC_Triangle {
    Vector3d p[3];
};

int Polygonise(MC_Gridcell grid, double isolevel, MC_Triangle* triangles);

bool ProcessVolumeCell(Volume* vol, int x, int y, int z, double iso, SimpleMesh* mesh);

void marchingCubes(Volume& vol, float isoLevel, SimpleMesh& mesh);

#endif // MARCHING_CUBES_H