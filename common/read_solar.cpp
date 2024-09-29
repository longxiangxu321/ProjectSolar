#include "read_solar.h"

namespace solarcity {

inline double degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0;
  }

std::vector<vec3f> readCSVandTransform(const std::string& filename) {
    std::vector<vec3f> unitVectors;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::ostringstream oss;
        oss << "Could not open file: " << filename;
        throw std::runtime_error(oss.str());
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        // Skip the first column (date/time)
        std::getline(ss, cell, ',');

        std::getline(ss, cell, ',');

        std::getline(ss, cell, ',');

        std::getline(ss, cell, ',');
        double apparent_elevation = std::stod(cell);


        // Skip elevation
        std::getline(ss, cell, ',');

        // Read azimuth
        std::getline(ss, cell, ',');
        double azimuth = std::stod(cell);


        // Skip remaining columns if any
        while (std::getline(ss, cell, ',')) {}
        
        double theta = degrees_to_radians(apparent_elevation);
        double phi =  degrees_to_radians(azimuth - 90);

        // Conversion to unit vector (in spherical coordinates)
        double x = cos(theta) * cos(phi);
        double y = - cos(theta) * sin(phi);
        double z = sin(theta);

        unitVectors.push_back(vec3f(x, y, z));
    }

    file.close();
    return unitVectors;
    }

    json readConfig(const std::string& filename) {
        std::ifstream file(filename);
        json CFG;
        if (file.is_open()) {
            try {
                file >> CFG;
            } catch (const json::parse_error& e) {
                std::cerr << "JSON parsing error for config file: " << e.what() << '\n';
                // Handle the error or throw an exception
            }
        } else {
            std::cerr << "Unable to open config file: " << filename << '\n';
            // Handle the error or throw an exception
        }
        return CFG;
    }

}