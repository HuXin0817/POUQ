# POUQ Makefile
# Supports only x86_64 architecture with AVX2 and FMA optimizations
# Requires Intel MKL library for high-performance sorting

# Compiler and flags
CC = gcc
CFLAGS = -std=c17 -Wall -Wextra -O3 -mavx2 -mfma -fopenmp -DMKL_ILP64 -I$(MKLROOT)/include
LDFLAGS = -lm -fopenmp -lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -L$(MKLROOT)/lib/intel64

# Directories
SRCDIR = libpouq
SIMDDIR = $(SRCDIR)/simd
OBJDIR = build
LIBDIR = $(OBJDIR)

# Auto-discover source files
LIB_SOURCES = $(shell find $(SRCDIR) -name "*.c")
EXAMPLE_SOURCES = example.c

# Object files
LIB_OBJECTS = $(LIB_SOURCES:%.c=$(OBJDIR)/%.o)
EXAMPLE_OBJECT = $(EXAMPLE_SOURCES:%.c=$(OBJDIR)/%.o)

# Targets
LIBRARY = $(LIBDIR)/libpouq.a
EXAMPLE = $(OBJDIR)/example

# Default target
all: $(EXAMPLE)

# Create build directories
$(OBJDIR):
	mkdir -p $(OBJDIR)/$(SRCDIR)
	mkdir -p $(OBJDIR)/$(SIMDDIR)

# Build library
$(LIBRARY): $(LIB_OBJECTS) | $(OBJDIR)
	ar rcs $@ $^

# Build example executable
$(EXAMPLE): $(EXAMPLE_OBJECT) $(LIBRARY) | $(OBJDIR)
	$(CC) $(EXAMPLE_OBJECT) -L$(LIBDIR) -lpouq $(LDFLAGS) -o $@

# Compile source files to object files
$(OBJDIR)/%.o: %.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR)

# Show discovered source files
sources:
	@echo "Discovered source files:"
	@echo "Library sources:"
	@echo "$(LIB_SOURCES)" | tr ' ' '\n' | sed 's/^/  /'
	@echo "Example sources:"
	@echo "$(EXAMPLE_SOURCES)" | tr ' ' '\n' | sed 's/^/  /'

# Install (optional)
install: $(EXAMPLE)
	@echo "Installing POUQ..."
	@echo "Library: $(LIBRARY)"
	@echo "Example: $(EXAMPLE)"
	@echo "Installation complete."

# Show build information
info:
	@echo "POUQ Build Information:"
	@echo "  Compiler: $(CC)"
	@echo "  C Flags: $(CFLAGS)"
	@echo "  Link Flags: $(LDFLAGS)"
	@echo "  Architecture: x86_64 (AVX2 + FMA)"
	@echo "  Library: $(LIBRARY)"
	@echo "  Example: $(EXAMPLE)"

# Run example
run: $(EXAMPLE)
	./$(EXAMPLE)

# Debug build
debug: CFLAGS = -std=c17 -Wall -Wextra -g -O0 -mavx2 -mfma -fopenmp -DDEBUG
debug: clean $(EXAMPLE)

# Help
help:
	@echo "POUQ Makefile Targets:"
	@echo "  all      - Build example executable (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  sources  - Show discovered source files"
	@echo "  install  - Show installation info"
	@echo "  info     - Show build information"
	@echo "  run      - Build and run example"
	@echo "  debug    - Build with debug symbols"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Note: POUQ requires x86_64 architecture with AVX2 and FMA support."
	@echo "      Intel MKL library is required. Set MKLROOT environment variable"
	@echo "      to MKL installation directory before building."

# Phony targets
.PHONY: all clean sources install info run debug help

# Auto-generate dependencies
-include $(LIB_OBJECTS:.o=.d)
-include $(EXAMPLE_OBJECT:.o=.d)

# Generate dependency files
$(OBJDIR)/%.d: %.c | $(OBJDIR)
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -MM -MT $(@:.d=.o) $< > $@
