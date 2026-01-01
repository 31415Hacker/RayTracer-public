#!/usr/bin/bash

echo "=================================================="
echo "            Compiling C++ test code               "
echo "=================================================="

echo ""

g++ -std=c++20 -O3 -march=native -flto tests/test.cpp -o bin/test
if [ $? -eq 0 ]; then
    echo "Compilation successful! The executable 'bin/test' has been created."
    echo "Done!"
else
    echo "Compilation failed. Please check the errors and see what went wrong."
fi