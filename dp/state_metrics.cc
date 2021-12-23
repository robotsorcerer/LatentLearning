#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <math.h>
#include <bits/stdc++.h>
#include <gvc.h>

using namespace std;

typedef vector<pair<string,float>> state_distribution;

const double epsilon = 1e-4;

void print_state_distribution(const state_distribution& sd)
{
  for(auto& sp : sd)
    cout << " " << sp.first << ":" << sp.second;
  cout << endl;
}

void parse_state_distribution(char* line, state_distribution& sd)
{
  char* state;
  double total = 0.;
  while ((state=strtok(nullptr, ":")) != nullptr)
    {
      char* probability = strtok(nullptr, " \t\n");
      if (probability != nullptr)
	sd.push_back(make_pair(state,atof(probability)));
      else
	sd.push_back(make_pair(state,1.));
      total += sd.back().second;
    }
  if (fabs(1. - total) > epsilon)
    {
      cerr << "bad state distribution: ";
      print_state_distribution(sd);
    }
}
  
struct marginal
{
  size_t count;
  state_distribution state_probability;
};

marginal parse_marginal(char* line)
{
  marginal ret;
  ret.count = atoi(strtok(line, " \t"));
  parse_state_distribution(line, ret.state_probability);
  return ret;
}

void print_marginal(const marginal& m)
{
  cout << m.count;
  print_state_distribution(m.state_probability);
  cout << endl;
}

struct action_transition
{
  string action;
  string state;
  state_distribution next_state_probability;
};

action_transition parse_action_transition(char* line)
{
  action_transition ret;
  ret.action = strtok(line, " \t");
  ret.state = strtok(nullptr, " \t");
  parse_state_distribution(line, ret.next_state_probability);
  return ret;
}

void print_action_transition(const action_transition& at)
{
  cout << at.action << " " << at.state;
  print_state_distribution(at.next_state_probability);
}

void print_action_transitions(const vector<action_transition>& ats)
{
  for (auto& at : ats)
    print_action_transition(at);
  cout << endl;
}

struct state_explanation
{
  string state;
  state_distribution alt_state_probability;
};

state_explanation parse_state_explanation(char* line)
{
  state_explanation ret;
  ret.state = strtok(line, " \t");
  parse_state_distribution(line, ret.alt_state_probability);
  return ret;
}

void print_state_explanation(const state_explanation& se)
{
  cout << se.state;
  print_state_distribution(se.alt_state_probability);
}

void print_state_explanations(const vector<state_explanation>& ses)
{
  for (auto& se : ses)
    print_state_explanation(se);
  cout << endl;
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

double impurity(vector<state_explanation>& ses)
{
  double total = 0.;
  for (auto& se : ses)
    {
      float max = 0.;
      for (auto& sp : se.alt_state_probability)
	{
	  if (sp.second > max)
	    max = sp.second;
	}
      total += 1. - max;
    }
  return total / ses.size();
}

void sss_and_dsm(vector<state_explanation>& learned_by_ground, vector<state_explanation>& ground_by_learned)
{ //sss = same_state_separated = uniform (over ground states) fraction not accounted for by a single learned state
  cout << "SSS (same state separated) = " << impurity(ground_by_learned) << endl;

  //dsm = different_state_marged = uniform (over learned states) fraction not accounted for by a single ground state
  cout << "DSM (different state merged) = " << impurity(learned_by_ground) << endl;
}

bool most_probable_first(pair<string,float>& e1, pair<string,float>& e2)
{
  return e1.second > e2.second;
}

state_distribution find_ground_states(const string& state, const vector<state_explanation>& learned_by_ground)
{
  for (auto& se : learned_by_ground)//this could be constant instead of linear
    if (se.state == state)
      return se.alt_state_probability;
  cerr << "badness! search for state '" << state << "' failed!" << endl;
  state_distribution ret;
  return ret;
}

void add_in(state_distribution& target, const state_distribution& source, float probability)
{//could be more efficient with a sortable state.
  for(auto& ssp : source)
    {
      bool not_added = true;
      for(auto& tsp : target)
	if (tsp.first == ssp.first)
	  {
	    tsp.second += ssp.second*probability;
	    not_added = false;
	    break;
	  }
      if (not_added)
	{
	  target.push_back(ssp);
	  target.back().second *= probability;
	}
    }
}

float difference(const state_distribution& sd1, const state_distribution& sd2)
{
  double total = 0.;
  for (auto& sp1 : sd1)
    {
      bool not_diffed = true;
      for (auto& sp2 : sd2)
	if (sp1.first == sp2.first)
	  {
	    total += fabs(sp1.second - sp2.second);
	    not_diffed = false;
	    break;
	  }
      if (not_diffed)
	total += sp1.second;
    }
  return total;
}

void transition_difference(vector<action_transition>& learned_transitions, vector<action_transition>& ground_transitions, vector<state_explanation>& learned_by_ground)
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
      for (auto& sp : at.next_state_probability)
	{// For each next learned state, find its distribution over ground states and add it in.
	  state_distribution ground_state_distribution = find_ground_states(sp.first, learned_by_ground);
	  add_in(p_learned_of_ground, ground_state_distribution, sp.second);
	}
      state_distribution ground_explanation_of_state = find_ground_states(at.state, learned_by_ground);
      
      state_distribution p_ground_of_ground;
      for (auto& sp : ground_explanation_of_state)
	{
	  bool not_added_in = true;
	  for (auto& at2 : ground_transitions) // for each ground state, find matching next state distribution
	    {
	      if (sp.first == at2.state && at.action == at2.action)
		{
		  add_in(p_ground_of_ground, at2.next_state_probability, sp.second);
		  not_added_in = false;
		  break;
		}
	    }
	  if (not_added_in)
	    cout << "badness, can't find match for " << sp.first << endl;
	}
      total += difference(p_learned_of_ground, p_ground_of_ground);
    }

  cout << "Dynamics difference = " << total / learned_transitions.size() << endl;
}

void graph_dynamics(vector<action_transition> learned_transitions)
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
      Agnode_t* state_node = agnode(g,(char*)ast.state.c_str(),1);
      /* Set an attribute - in this case one that affects the visible rendering */
      agsafeset(state_node, (char*)"color", (char*)"red", (char*)"red");
      agsafeset(state_node, (char*)"fontsize", (char*)"8", (char*)"8");
      agsafeset(state_node, (char*)"height", (char*)"0.25", (char*)"0.25");
      agsafeset(state_node, (char*)"width", (char*)"0.25", (char*)"0.25");
      string state_action = ast.state+"_"+ast.action;
      Agnode_t* state_action_node = agnode(g,(char*)state_action.c_str(),1);
      agsafeset(state_action_node, (char*)"color", (char*)"green", (char*)"");
      Agedge_t* e = agedge(g,state_node, state_action_node, 0, 1);
      agsafeset(e, (char*)"len", (char*)"0.5", (char*)"0.5");
      agsafeset(e, (char*)"style", (char*)"filled", (char*)"filled");
      for (auto& sp : ast.next_state_probability)
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
  vector<action_transition> learned_transitions;
  vector<action_transition> ground_transitions;
  vector<state_explanation> learned_by_ground;
  vector<state_explanation> ground_by_learned;
  
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
	learned_transitions.push_back(parse_action_transition(line));
	break;
      case 2: // ground state occupancy
	ground_marginals = parse_marginal(line);
	break;
      case 3: // ground state transition
	ground_transitions.push_back(parse_action_transition(line));
	break;
      case 4: // learned state explanation
	learned_by_ground.push_back(parse_state_explanation(line));
	break;
      case 5: // ground state explanation
	ground_by_learned.push_back(parse_state_explanation(line));
	break;
      default:
	cerr << "something wrong, unknown table" << endl;
      }
  }
  free(line);
  fclose(stream);
  /*
  cout << "#read in:" << endl;
  cout << "#learned_marginals" << endl;
  print_marginal(learned_marginals);
  cout << "#learned_transitions" << endl;
  print_action_transitions(learned_transitions);
  cout << "#ground_marginals" << endl;
  print_marginal(ground_marginals);
  cout << "#ground_transitions" << endl;
  print_action_transitions(ground_transitions);
  cout << "#learned_by_ground" << endl;
  print_state_explanations(learned_by_ground);
  cout << "#ground_by_learned" << endl;
  print_state_explanations(ground_by_learned);
  */
  //learned-only metrics
  cout << "learned state ";
  min_entropy(learned_marginals);
  graph_dynamics(learned_transitions);
  
  //learned+ground metrics
  cout << "ground state ";
  min_entropy(ground_marginals);
  sss_and_dsm(learned_by_ground, ground_by_learned);
  transition_difference(learned_transitions, ground_transitions, learned_by_ground);
  
  exit(EXIT_SUCCESS);
}
