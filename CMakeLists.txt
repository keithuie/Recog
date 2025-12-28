cmake_minimum_required(VERSION 3.14)
project(MachineIQ_Host VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---

# 1. Python & pybind11 (For the Model)
# This is critical for the "Zero-Copy" embedding.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 2. nlohmann/json (For Parsing configuration/results)
find_package(nlohmann_json REQUIRED)

# 3. Paho MQTT (Optional/Placeholder)
# Dependency removed to simplify build on macOS
# find_package(EclipsePahoMqttCpp REQUIRED)

# 4. CURL (For API Ingest like PureSignal/Sensoteq)
find_package(CURL REQUIRED)

# 5. SQL Drivers (Placeholder for Alta Solutions)
# In production, you would add: find_package(ODBC) or find_package(PostgreSQL)
# We will use standard libraries to mock the connection logic in main.cpp for now.

# --- Target Definition ---

add_executable(machineiq_host main.cpp)

# Link Libraries
target_link_libraries(machineiq_host
  PRIVATE
    pybind11::embed
    nlohmann_json::nlohmann_json
    CURL::libcurl
)

target_include_directories(machineiq_host PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Post-build: Ensure python source is near the executable
add_custom_command(TARGET machineiq_host POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/machine_iq_core.py
    $<TARGET_FILE_DIR:machineiq_host>/machine_iq_core.py
)
