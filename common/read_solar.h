#pragma once

#include <string>
#include <vector>
#include "gdt/math/AffineSpace.h"
#include <filesystem>
#include <fstream> 
#include <vector>
#include <sstream>
#include "3rdParty/json.hpp"

using json = nlohmann::json;

namespace solarcity {

    using namespace gdt;

    std::vector<vec3f> readCSVandTransform(const std::string& filename);

    json readConfig(const std::string& filename);

}