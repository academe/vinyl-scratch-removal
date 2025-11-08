# Makefile for Vinyl Scratch Removal LV2 plugin

# Plugin name and version
PLUGIN_NAME = vinyl_scratch_removal
VERSION = 1.0.0

# Build configuration
CC = gcc
CFLAGS = -O3 -ffast-math -fPIC -Wall -Wextra
LDFLAGS = -shared -lm

# Installation paths
PREFIX ?= /usr/local
LV2_DIR = $(PREFIX)/lib/lv2
BUNDLE_DIR = $(PLUGIN_NAME).lv2

# Source files
SRC = vinyl_scratch_removal.c
OBJ = $(SRC:.c=.o)
SO = $(PLUGIN_NAME).so

# Targets
.PHONY: all clean install uninstall

all: $(BUNDLE_DIR)/$(SO) $(BUNDLE_DIR)/manifest.ttl $(BUNDLE_DIR)/$(PLUGIN_NAME).ttl

$(BUNDLE_DIR):
	mkdir -p $(BUNDLE_DIR)

$(BUNDLE_DIR)/$(SO): $(OBJ) | $(BUNDLE_DIR)
	$(CC) $(LDFLAGS) -o $@ $^

$(BUNDLE_DIR)/manifest.ttl: manifest.ttl | $(BUNDLE_DIR)
	cp $< $@

$(BUNDLE_DIR)/$(PLUGIN_NAME).ttl: $(PLUGIN_NAME).ttl | $(BUNDLE_DIR)
	cp $< $@

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ)
	rm -rf $(BUNDLE_DIR)

install: all
	mkdir -p $(LV2_DIR)
	cp -r $(BUNDLE_DIR) $(LV2_DIR)/

uninstall:
	rm -rf $(LV2_DIR)/$(BUNDLE_DIR)

# Help
help:
	@echo "Vinyl Scratch Removal LV2 Plugin"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build plugin bundle (default)"
	@echo "  clean     - Remove build files"
	@echo "  install   - Install plugin to $(LV2_DIR)"
	@echo "  uninstall - Remove installed plugin"
	@echo ""
	@echo "Variables:"
	@echo "  PREFIX    - Installation prefix (default: /usr/local)"
	@echo "  LV2_DIR   - LV2 plugin directory (default: PREFIX/lib/lv2)"
