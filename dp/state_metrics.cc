#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <math.h>

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

// max_elements log (1/probability)
void min_entropy(const marginal& m)
{
  float min_value = 0.;
  for (auto& sp : m.state_probability)
    if (log(1. / sp.second) > min_value)
      min_value = log(1. / sp.second);
  cout << "minimum entropy = " << min_value << endl;
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
  min_entropy(learned_marginals);

  //learned+ground metrics
  sss_and_dsm(learned_by_ground, ground_by_learned);
  
  exit(EXIT_SUCCESS);
}
