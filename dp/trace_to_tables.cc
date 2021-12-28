//read in a trace and output tables for state metrics
#include <iostream>
#include <string.h>
#include "state_tables.h"

using namespace std;

void print_state_distribution(const state_distribution& sd)
{
  for(auto& sp : sd)
    cout << " " << sp.first << ":" << sp.second;
  cout << endl;
}

void print_marginal(const marginal& m)
{
  cout << m.count;
  print_state_distribution(m.state_probability);
  cout << endl;
}

void print_action_transition(const pair<action_state,state_distribution>& at)
{
  cout << at.first.action << " " << at.first.state;
  print_state_distribution(at.second);
}

void print_action_transitions(const action_transitions& ats)
{
  for (auto& at : ats)
    print_action_transition(at);
  cout << endl;
}

void print_state_explanations(const state_explanations& ses)
{
  for (auto& se : ses)
    {
      cout << se.first;
      print_state_distribution(se.second);
    }
  cout << endl;
}

void print_state_examples(state_examples& ses)
{
  for (auto& se : ses)
    {
      cout << se.first;
      for (auto& uri : se.second)
	cout << " " << uri;
      cout << endl;  
    }
  cout << endl;  
}

size_t normalize_state_distribution(state_distribution& sd)
{
  double total = 0;
  for (auto& sp : sd)
    total += sp.second;

  double multiplier = 1. / total;
  for (auto& sp : sd)
    sp.second *= multiplier;

  return total;
}

void increment_state_probability(string& s, state_distribution& sd)
{
  if (sd.find(s) != sd.end())
    ++sd[s];
  else
    sd.emplace(s,1.);
}

template<class T> void ensure_state_distribution(T& key, unordered_map<T,state_distribution>& map)
{
  if (map.find(key) == map.end())
    {
      state_distribution sd;
      map.emplace(key,sd);
    }
}

int main(int argc, char* argv[])
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

  while ((nread = getline(&line, &len, stream)) != -1)
    {
      if (*line == '#' || *line == '\n')
	continue;

      string learned;
      string ground;
      string action;
      string next_learned;
      string next_ground;
      string uri;
      
      char* lineptr = line;
      char* key;
      while ((key=strtok(lineptr, ":")) != nullptr)
	{
	  string skey = key;
	  string value = strtok(nullptr, " \t\n");
	  lineptr = nullptr;
	  if (skey == "learned")
	    learned=value;
	  else if (skey == "ground")
	    ground=value;
	  else if (skey == "action")
	    action=value;
	  else if (skey == "next_learned")
	    next_learned=value;
	  else if (skey == "next_ground")
	    next_ground=value;
	  else if (skey == "uri")
	    uri=value;	  
	}

      if (learned != "")
	{
	  increment_state_probability(learned, learned_marginals.state_probability);
	  if (action != "" && next_learned != "")
	    {
	      action_state as = {action, learned};
	      ensure_state_distribution(as,learned_transitions);
	      increment_state_probability(next_learned, learned_transitions[as]);
	    }
	}
      if (ground != "")
	{
	  increment_state_probability(ground, ground_marginals.state_probability);
	  if (action != "" && next_ground != "")
	    {
	      action_state as = {action, ground};
	      ensure_state_distribution(as,ground_transitions);
	      increment_state_probability(next_ground, ground_transitions[as]);
	    }
	}
      if (learned != "" && ground != "")
	{
	  ensure_state_distribution(learned,learned_by_ground);
	  increment_state_probability(ground, learned_by_ground[learned]);
	  ensure_state_distribution(ground,ground_by_learned);
	  increment_state_probability(learned, ground_by_learned[ground]);
	}
      if (uri != "")
	{
	  if (learned != "")
	    {
	      if (learned_examples.find(learned) == learned_examples.end())
		{
		  vector<string> vs;
		  learned_examples.emplace(learned,vs);
		}
	      learned_examples[learned].push_back(uri);
	    }
	  if (ground != "")
	    {
	      if (ground_examples.find(ground) == ground_examples.end())
		{
		  vector<string> vs;
		  ground_examples.emplace(ground,vs);
		}
	      ground_examples[ground].push_back(uri);
	    }
	}
    }

  
  cout << "#read in:" << endl;
  cout << "#learned_marginals" << endl;
  learned_marginals.count = normalize_state_distribution(learned_marginals.state_probability);
  print_marginal(learned_marginals);
  cout << "#learned_transitions" << endl;
  for (auto& trans : learned_transitions)
    normalize_state_distribution(trans.second);
  print_action_transitions(learned_transitions);
  cout << "#learned_examples" << endl;
  if (learned_examples.size() > 0)
    print_state_examples(learned_examples);
  else
    cout << "no_examples\n" << endl;
  cout << "#ground_marginals" << endl;
  ground_marginals.count = normalize_state_distribution(ground_marginals.state_probability);
  print_marginal(ground_marginals);
  cout << "#ground_transitions" << endl;
  for (auto& trans : ground_transitions)
    normalize_state_distribution(trans.second);
  print_action_transitions(ground_transitions);
  cout << "#ground_examples" << endl;
  if (ground_examples.size() > 0)
    print_state_examples(ground_examples);
  else
    cout << "no_examples\n" << endl;
  cout << "#learned_by_ground" << endl;
  for (auto& trans : learned_by_ground)
    normalize_state_distribution(trans.second);
  print_state_explanations(learned_by_ground);
  cout << "#ground_by_learned" << endl;
  for (auto& trans : ground_by_learned)
    normalize_state_distribution(trans.second);
  print_state_explanations(ground_by_learned);
}
