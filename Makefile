# POUQ Makefile
# Supports only x86_64 architecture with AVX2 and FMA optimizations

# Compiler and flags
CC = gcc
CFLAGS = -std=c2x -Wall -Wextra -O3 -mavx2 -mfma -fopenmp
LDFLAGS = -lm -fopenmp -lpthread

# Directories
SRCDIR = libpouq
OBJDIR = build
LIBDIR = $(OBJDIR)

# Auto-discover source files
LIB_SOURCES = $(shell find $(SRCDIR) -name "*.c")
LIB_HEADERS = $(shell find $(SRCDIR) -name "*.h")
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
	./$(EXAMPLE) $(ARGS)

# Debug build
debug: CFLAGS = -std=c17 -Wall -Wextra -g -O0 -mavx2 -mfma -fopenmp -DDEBUG
debug: clean $(EXAMPLE)

# Format source code using clang-format
FMT_TOOL = clang-format
SRC_FILES = $(LIB_SOURCES) $(EXAMPLE_SOURCES)

fmt:
	@echo "Formatting source files..."
	$(FMT_TOOL) -i $(SRC_FILES) $(LIB_HEADERS)
	@echo "Formatting complete."

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
	@echo "  fmt      - Format source code with clang-format"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Note: POUQ requires x86_64 architecture with AVX2 and FMA support."

# Phony targets
.PHONY: all clean sources install info run debug fmt help

# Auto-generate dependencies
-include $(LIB_OBJECTS:.o=.d)
-include $(EXAMPLE_OBJECT:.o=.d)

# Generate dependency files
$(OBJDIR)/%.d: %.c | $(OBJDIR)
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -MM -MT $(@:.d=.o) $< > $@
