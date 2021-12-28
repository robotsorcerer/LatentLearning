#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <bits/stdc++.h>
#include <gvc.h>
#include "state_tables.h"

using namespace std;

const double epsilon = 1e-4;

void parse_state_distribution(char* line, state_distribution& sd)
{
  char* state;
  double total = 0.;
  while ((state=strtok(nullptr, ":")) != nullptr)
    {
      char* probability = strtok(nullptr, " \t\n");
      if (probability != nullptr)
	sd.emplace(state,atof(probability));
      else
	sd.emplace(state,1.);
      total += sd[state];
    }
  if (fabs(1. - total) > epsilon)
    cerr << "bad state distribution" << endl;
}
  
marginal parse_marginal(char* line)
{
  marginal ret;
  ret.count = atoi(strtok(line, " \t"));
  parse_state_distribution(line, ret.state_probability);
  return ret;
}

void parse_action_transition(action_transitions& ats, char* line)
{
  action_state as;
  as.action = strtok(line, " \t");
  as.state = strtok(nullptr, " \t");
  state_distribution sd;
  parse_state_distribution(line, sd);
  ats.emplace(as,sd);
}

void parse_state_explanation(state_explanations& se, char* line)
{
  string key = strtok(line, " \t");
  state_distribution sd;
  parse_state_distribution(line, sd);
  se.emplace(key,sd);
}

void parse_state_examples(state_examples& se, char* line)
{
  string key = strtok(line, " \t");

  vector<string> value;
  char* uri;
  while ((uri=strtok(nullptr, " \t\n")) != nullptr)
    value.push_back(uri);

  se.emplace(key,value);
}

// max_elements log (1/probability)
void min_entropy(const marginal& m)
{
  float min_value = 0.;
  for (auto& sp : m.state_probability)
    if (log(1. / sp.second) > min_value)
      min_value = log(1. / sp.second);
  cout << "minimum entropy = " << min_value << endl;
}

double impurity(state_explanations& ses)
{
  double total = 0.;
  for (auto& se : ses)
    {
      float max = 0.;
      for (auto& sp : se.second)
	{
	  if (sp.second > max)
	    max = sp.second;
	}
      total += 1. - max;
    }
  return total / ses.size();
}

void sss_and_dsm(state_explanations& learned_by_ground, state_explanations& ground_by_learned)
{ //sss = same_state_separated = uniform (over ground states) fraction not accounted for by a single learned state
  cout << "SSS (same state separated) = " << impurity(ground_by_learned) << endl;

  //dsm = different_state_marged = uniform (over learned states) fraction not accounted for by a single ground state
  cout << "DSM (different state merged) = " << impurity(learned_by_ground) << endl;
}

bool most_probable_first(pair<string,float>& e1, pair<string,float>& e2)
{
  return e1.second > e2.second;
}

void add_in(state_distribution& target, const state_distribution& source, float probability)
{//could be more efficient with a sortable state.
  for(auto& ssp : source)
    {
      float p = ssp.second * probability;
      auto te = target.find(ssp.first);
      if (te != target.end())
	te->second += p;
      else
	target.emplace(ssp.first,p);
    }
}

float difference(const state_distribution& sd1, const state_distribution& sd2)
{
  double total = 0.;
  for (auto& sp1 : sd1)
    {
      auto sd2i = sd2.find(sp1.first);
      if (sd2i != sd2.end())
	total += fabs(sp1.second - sd2i->second);
      else
	total += sp1.second;
    }
  return total;
}

void normalized(const state_distribution& sd)
{
  double total = 0.;
  for (auto& sp : sd)
    total+=sp.second;
  if (fabs(total - 1.) > epsilon)
    cerr << "normalization failure!" << endl;
}

void transition_difference(action_transitions& learned_transitions, action_transitions& ground_transitions, state_explanations& learned_by_ground)
{
  //Every learned state induces a distribution over ground states via learned_by_ground.
  //For each learned state action, we have a distribution over learned next states.
  //The distribution over next states can be translated into a distribution over ground states using
  //learned_by_ground.  Doing that provides a p_learned (s'_ground | s_learned,a).
  //Alternatively, we can look at E_{s_ground ~ s_learned} p_ground(s'_ground | s_ground, a).
  //This gives us two distributions over s'_ground, which we can take the l_1 difference between.
  //And then take the uniform average over all s_learned,a.
  double total = 0.;
  for (auto& at : learned_transitions)
    {
      state_distribution p_learned_of_ground;
      for (auto& sp : at.second) // For each next learned state, find its distribution over ground states and add it in.
	add_in(p_learned_of_ground, learned_by_ground[sp.first], sp.second);
      
      state_distribution ground_explanation_of_state = learned_by_ground[at.first.state];
      
      state_distribution p_ground_of_ground;
      for (auto& sp : ground_explanation_of_state) // For the state, apply its distribution over ground states to the next state distribution.
	{
	  action_state as = {at.first.action, sp.first};
	  add_in(p_ground_of_ground, ground_transitions[as], sp.second);
	}
      total += difference(p_learned_of_ground, p_ground_of_ground);
    }

  cout << "Dynamics difference = " << total / learned_transitions.size() << endl;
}

void graph_dynamics(action_transitions& learned_transitions)
{//see here: https://graphviz.org/pdf/libguide.pdf for details
  Agraph_t *g;
  GVC_t *gvc;
  /* set up a graphviz context */
  gvc = gvContext();
  /* parse command line args - minimally argv[0] sets layout engine */
  char* args[] = { (char*)"neato", (char*)"-Tgif", (char*)"-ovisualization.gif" };
  gvParseArgs(gvc, sizeof(args)/sizeof(char*), args);
  /* Create a simple digraph */
  g = agopen((char*)"g", Agdirected, nullptr);
  //  agsafeset(g, (char*)"nodesep", (char*)"0.75", (char*)"0.75");
  vector<Agnode_t*> nodes;
  for (auto& ast : learned_transitions)
    {
      Agnode_t* state_node = agnode(g,(char*)ast.first.state.c_str(),1);
      /* Set an attribute - in this case one that affects the visible rendering */
      agsafeset(state_node, (char*)"color", (char*)"red", (char*)"red");
      agsafeset(state_node, (char*)"fontsize", (char*)"8", (char*)"8");
      agsafeset(state_node, (char*)"height", (char*)"0.25", (char*)"0.25");
      agsafeset(state_node, (char*)"width", (char*)"0.25", (char*)"0.25");
      string state_action = ast.first.state+"_"+ast.first.action;
      Agnode_t* state_action_node = agnode(g,(char*)state_action.c_str(),1);
      agsafeset(state_action_node, (char*)"color", (char*)"green", (char*)"");
      Agedge_t* e = agedge(g,state_node, state_action_node, 0, 1);
      agsafeset(e, (char*)"len", (char*)"0.5", (char*)"0.5");
      agsafeset(e, (char*)"style", (char*)"filled", (char*)"filled");
      for (auto& sp : ast.second)
	{
	  Agnode_t* next_state_node = agnode(g,(char*)sp.first.c_str(),1);
	  Agedge_t* e = agedge(g,state_action_node,next_state_node,nullptr,1);
	  agsafeset(e, (char*)"len", (char*)"1.0", (char*)"1.0");
	  agsafeset(e, (char*)"style", (char*)"dashed", (char*)"dashed");
	}
    }
  /* Compute a layout using layout engine from command line args */
  gvLayoutJobs(gvc, g);
  /* Write the graph according to -T and -o options */
  gvRenderJobs(gvc, g);
  /* Free layout data */
  gvFreeLayout(gvc, g);
  /* Free graph structures */
  agclose(g);
  /* close output file, free context, and return number of errors */
  gvFreeContext(gvc);
}

int main(int argc, char *argv[])
{
  FILE *stream;
  char *line = NULL;
  size_t len = 0;
  ssize_t nread;
  
  if (argc > 2) {
    fprintf(stderr, "Usage: %s <file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  else if (argc == 1)
    stream = stdin;
  else 
    stream = fopen(argv[1], "r");
  if (stream == NULL) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  marginal learned_marginals;
  marginal ground_marginals;
  action_transitions learned_transitions;
  action_transitions ground_transitions;
  state_explanations learned_by_ground;
  state_explanations ground_by_learned;
  state_examples learned_examples;
  state_examples ground_examples;
  
  size_t table = 0;
  bool mid_table = false;
  while ((nread = getline(&line, &len, stream)) != -1) {
    if (*line == '#')//a comment
      continue;

    if (*line == '\n')//a newline
      {
	if (mid_table)//table finishes
	  {
	    ++table;
	    mid_table = false;
	  }
	continue;
      }

    mid_table = true;

    switch (table)
      {
      case 0: // learned state occupancy
	learned_marginals = parse_marginal(line);
	break;
      case 1: // learned state transition
	parse_action_transition(learned_transitions,line);
	break;
      case 2: // learned_state_examples
	parse_state_examples(learned_examples,line);
	break;
      case 3: // ground state occupancy
	ground_marginals = parse_marginal(line);
	break;
      case 4: // ground state transition
	parse_action_transition(ground_transitions,line);
	break;
      case 5: // ground state examples
	parse_state_examples(ground_examples,line);
	break;
      case 6: // learned state explanation
	parse_state_explanation(learned_by_ground,line);
	break;
      case 7: // ground state explanation
	parse_state_explanation(ground_by_learned,line);
	break;
      default:
	cerr << "something wrong, unknown table" << endl;
      }
  }
  free(line);
  fclose(stream);

  //learned-only metrics
  cout << learned_marginals.state_probability.size() << " learned states ";
  min_entropy(learned_marginals);
  graph_dynamics(learned_transitions);
  
  //learned+ground metrics
  cout << ground_marginals.state_probability.size() << " ground state ";
  min_entropy(ground_marginals);
  sss_and_dsm(learned_by_ground, ground_by_learned);
  transition_difference(learned_transitions, ground_transitions, learned_by_ground);
  
  exit(EXIT_SUCCESS);
}
