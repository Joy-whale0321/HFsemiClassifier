# Makefile for ppHF_eXDecay.cc

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

# From config scripts
PYTHIA8_CFLAGS := $(shell pythia8-config --cxxflags)
PYTHIA8_LIBS   := $(shell pythia8-config --libs)

FASTJET_CFLAGS := $(shell fastjet-config --cxxflags)
FASTJET_LIBS   := $(shell fastjet-config --libs)

ROOT_CFLAGS    := $(shell root-config --cflags)
ROOT_LIBS      := $(shell root-config --libs)

INCLUDES := $(PYTHIA8_CFLAGS) $(FASTJET_CFLAGS) $(ROOT_CFLAGS)
LIBS     := $(PYTHIA8_LIBS) $(FASTJET_LIBS) $(ROOT_LIBS)

TARGET := ppHF_eXDecay
SRC    := ppHF_eXDecay.cc
OBJ    := $(SRC:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
