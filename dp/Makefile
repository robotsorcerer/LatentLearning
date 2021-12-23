#sudo apt install libgraphviz-dev
CFLAGS= $(shell pkg-config libgvc --cflags) -Wall -g -O2 -std=c++11 -fno-stack-protector
LDFLAGS= $(shell pkg-config libgvc --libs)

state_metrics: state_metrics.cc
	g++ state_metrics.cc $(CFLAGS) $(LDFLAGS) -o $@
