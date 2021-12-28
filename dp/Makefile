#sudo apt install libgraphviz-dev
CFLAGS= $(shell pkg-config libgvc --cflags) -Wall -g -O2 -std=c++11 -fno-stack-protector
LDFLAGS= $(shell pkg-config libgvc --libs)

all: state_metrics trace_to_tables

state_metrics: state_metrics.cc state_tables.h
	g++ state_metrics.cc $(CFLAGS) $(LDFLAGS) -o $@

trace_to_tables: trace_to_tables.cc state_tables.h
	g++ trace_to_tables.cc $(CFLAGS) $(LDFLAGS) -o $@
