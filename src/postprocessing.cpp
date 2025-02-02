#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_set>
#include <queue>
#include <eigen3/Eigen/Dense>

struct Vertex {
    double x, y, z;
};

struct Face {
    int v1, v2, v3;
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<Face> faces;

    bool loadOFF(const std::string& filename);
    bool saveOFF(const std::string& filename);
    void laplacianSmooth(int iterations, double lambda, double mu);
    void removeSmallComponents(int minSize);
};

bool Mesh::loadOFF(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(infile, line);
    if (line != "OFF") {
        std::cerr << "Not a valid OFF file: " << filename << std::endl;
        return false;
    }

    int numVertices, numFaces, numEdges;
    infile >> numVertices >> numFaces >> numEdges;

    vertices.resize(numVertices);
    faces.resize(numFaces);

    for (int i = 0; i < numVertices; ++i) {
        infile >> vertices[i].x >> vertices[i].y >> vertices[i].z;
    }

    for (int i = 0; i < numFaces; ++i) {
        int n, v1, v2, v3;
        infile >> n >> v1 >> v2 >> v3;
        if (n != 3) {
            std::cerr << "Only triangular faces are supported." << std::endl;
            return false;
        }
        faces[i] = {v1, v2, v3};
    }

    infile.close();
    return true;
}

bool Mesh::saveOFF(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    outfile << "OFF\n";
    outfile << vertices.size() << " " << faces.size() << " 0\n";

    for (const auto& vertex : vertices) {
        outfile << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    for (const auto& face : faces) {
        outfile << "3 " << face.v1 << " " << face.v2 << " " << face.v3 << "\n";
    }

    outfile.close();
    return true;
}

void Mesh::laplacianSmooth(int iterations, double lambda, double mu) {
    std::vector<Vertex> newVertices = vertices;

    for (int iter = 0; iter < iterations; ++iter) {
        // Laplacian smoothing step
        for (size_t i = 0; i < vertices.size(); ++i) {
            Eigen::Vector3d sum(0, 0, 0);
            int count = 0;

            for (const auto& face : faces) {
                if (face.v1 == i || face.v2 == i || face.v3 == i) {
                    if (face.v1 != i) {
                        sum += Eigen::Vector3d(vertices[face.v1].x, vertices[face.v1].y, vertices[face.v1].z);
                        ++count;
                    }
                    if (face.v2 != i) {
                        sum += Eigen::Vector3d(vertices[face.v2].x, vertices[face.v2].y, vertices[face.v2].z);
                        ++count;
                    }
                    if (face.v3 != i) {
                        sum += Eigen::Vector3d(vertices[face.v3].x, vertices[face.v3].y, vertices[face.v3].z);
                        ++count;
                    }
                }
            }

            if (count > 0) {
                Eigen::Vector3d avg = sum / count;
                newVertices[i].x = vertices[i].x + lambda * (avg.x() - vertices[i].x);
                newVertices[i].y = vertices[i].y + lambda * (avg.y() - vertices[i].y);
                newVertices[i].z = vertices[i].z + lambda * (avg.z() - vertices[i].z);
            }
        }
        vertices = newVertices;

        // Taubin smoothing step (negative Laplacian smoothing)
        for (size_t i = 0; i < vertices.size(); ++i) {
            Eigen::Vector3d sum(0, 0, 0);
            int count = 0;

            for (const auto& face : faces) {
                if (face.v1 == i || face.v2 == i || face.v3 == i) {
                    if (face.v1 != i) {
                        sum += Eigen::Vector3d(vertices[face.v1].x, vertices[face.v1].y, vertices[face.v1].z);
                        ++count;
                    }
                    if (face.v2 != i) {
                        sum += Eigen::Vector3d(vertices[face.v2].x, vertices[face.v2].y, vertices[face.v2].z);
                        ++count;
                    }
                    if (face.v3 != i) {
                        sum += Eigen::Vector3d(vertices[face.v3].x, vertices[face.v3].y, vertices[face.v3].z);
                        ++count;
                    }
                }
            }

            if (count > 0) {
                Eigen::Vector3d avg = sum / count;
                newVertices[i].x = vertices[i].x + mu * (avg.x() - vertices[i].x);
                newVertices[i].y = vertices[i].y + mu * (avg.y() - vertices[i].y);
                newVertices[i].z = vertices[i].z + mu * (avg.z() - vertices[i].z);
            }
        }
        vertices = newVertices;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.off> <output.off>" << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    Mesh mesh;
    if (!mesh.loadOFF(inputFilename)) {
        return 1;
    }

    mesh.laplacianSmooth(10, 0.09, -0.11);

    if (!mesh.saveOFF(outputFilename)) {
        return 1;
    }

    std::cout << "Postprocessing completed and saved to " << outputFilename << std::endl;
    return 0;
}