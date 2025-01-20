#include "MarchingCubes.h"
#include "SimpleMesh.h"
#include "Volume.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

void marchingCubes(Volume& vol, float isoLevel, SimpleMesh& mesh) {
    int gridSizeX = vol.getDimX();
    int gridSizeY = vol.getDimY();
    int gridSizeZ = vol.getDimZ();

    for (int x = 0; x < gridSizeX - 1; x++) {
        for (int y = 0; y < gridSizeY - 1; y++) {
            for (int z = 0; z < gridSizeZ - 1; z++) {
                ProcessVolumeCell(&vol, x, y, z, isoLevel, &mesh);
            }
        }
    }
}