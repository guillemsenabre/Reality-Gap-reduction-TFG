# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kyu8/tfg/rwork/src/dual_arm_control_pkg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kyu8/tfg/rwork/build/dual_arm_control_pkg

# Include any dependencies generated for this target.
include CMakeFiles/test_arduino.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_arduino.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_arduino.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_arduino.dir/flags.make

CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o: CMakeFiles/test_arduino.dir/flags.make
CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o: /home/kyu8/tfg/rwork/src/dual_arm_control_pkg/src/test_arduino.cpp
CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o: CMakeFiles/test_arduino.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kyu8/tfg/rwork/build/dual_arm_control_pkg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o -MF CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o.d -o CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o -c /home/kyu8/tfg/rwork/src/dual_arm_control_pkg/src/test_arduino.cpp

CMakeFiles/test_arduino.dir/src/test_arduino.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_arduino.dir/src/test_arduino.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kyu8/tfg/rwork/src/dual_arm_control_pkg/src/test_arduino.cpp > CMakeFiles/test_arduino.dir/src/test_arduino.cpp.i

CMakeFiles/test_arduino.dir/src/test_arduino.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_arduino.dir/src/test_arduino.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kyu8/tfg/rwork/src/dual_arm_control_pkg/src/test_arduino.cpp -o CMakeFiles/test_arduino.dir/src/test_arduino.cpp.s

# Object files for target test_arduino
test_arduino_OBJECTS = \
"CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o"

# External object files for target test_arduino
test_arduino_EXTERNAL_OBJECTS =

test_arduino: CMakeFiles/test_arduino.dir/src/test_arduino.cpp.o
test_arduino: CMakeFiles/test_arduino.dir/build.make
test_arduino: CMakeFiles/test_arduino.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kyu8/tfg/rwork/build/dual_arm_control_pkg/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_arduino"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_arduino.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_arduino.dir/build: test_arduino
.PHONY : CMakeFiles/test_arduino.dir/build

CMakeFiles/test_arduino.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_arduino.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_arduino.dir/clean

CMakeFiles/test_arduino.dir/depend:
	cd /home/kyu8/tfg/rwork/build/dual_arm_control_pkg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kyu8/tfg/rwork/src/dual_arm_control_pkg /home/kyu8/tfg/rwork/src/dual_arm_control_pkg /home/kyu8/tfg/rwork/build/dual_arm_control_pkg /home/kyu8/tfg/rwork/build/dual_arm_control_pkg /home/kyu8/tfg/rwork/build/dual_arm_control_pkg/CMakeFiles/test_arduino.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_arduino.dir/depend

