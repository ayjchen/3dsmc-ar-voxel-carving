#include "MarchingCubes.h"
#include "SimpleMesh.h"
#include "Volume.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

void marchingCubes(Volume& scalarField, float isoLevel, SimpleMesh& mesh) {
    int gridSizeX = scalarField.getDimX();
    int gridSizeY = scalarField.getDimY();
    int gridSizeZ = scalarField.getDimZ();

    for (int x = 0; x < gridSizeX - 1; x++) {
        for (int y = 0; y < gridSizeY - 1; y++) {
            for (int z = 0; z < gridSizeZ - 1; z++) {
                ProcessVolumeCell(&scalarField, x, y, z, isoLevel, &mesh);
            }
        }
    }
}