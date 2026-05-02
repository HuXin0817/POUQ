#!/bin/bash

git submodule update --init --recursive
cmake -B build -S . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=${1:-Release}
ln -sf build/compile_commands.json compile_commands.json
cmake --build build -j $(nproc) --target pypouq
