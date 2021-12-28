#pragma once
#include <string>
#include <unordered_map>
#include <vector>


typedef std::unordered_map<std::string,float> state_distribution;

struct marginal
{
  size_t count;
  state_distribution state_probability;
};

struct action_state
{
  std::string action;
  std::string state;

  bool operator==(const action_state &other) const
  { return (action == other.action
            && state == other.state);
  }
};

namespace std {
template <>
struct hash<action_state>
{
  size_t operator()(const action_state& k) const
  {
    return ((hash<string>()(k.action)
	     ^ (hash<string>()(k.state) << 1)) >> 1);
  }
};
}

typedef std::unordered_map<action_state,state_distribution> action_transitions;

typedef std::unordered_map<std::string,state_distribution> state_explanations;

typedef std::unordered_map<std::string,std::vector<std::string>> state_examples;
