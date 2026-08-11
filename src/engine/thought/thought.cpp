#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
const int MAX_AGENTS = 64000;
const int MAX_CONNECTIONS = 80000;
const int CONNECTION_RECORD_SIZE = 40;
const int DEFAULT_MAX_HOPS = 12;
struct AgentPos {
  float x, y, z;
  float vx, vy, vz;
};
struct LC {
  int idxA;
  int type;
  int idxB;
  float utility;
  int subjectQuantExact;
  int subjectTenseExact;
  int subjectTruthExact;
  int predicateQuantExact;
  int predicateTenseExact;
  int predicateTruthExact;
};
struct Edge {
  int connectionIndex;
  int from;
  int to;
  int reversed;
};
struct ReverseEdge {
  int connectionIndex;
  int from;
  int to;
  int reversed;
};
struct Seed {
  int agentIndex;
  int spawnCount;
  int goalIndex;
  int routeIndex;
};
struct Step {
  int connectionIndex;
  int from;
  int to;
  int reversed;
};
struct GoalRouteTree {
  std::vector<unsigned char> reachable;
  std::vector<int> nextNode;
  std::vector<int> nextConnection;
  std::vector<int> nextReversed;
};
static_assert(sizeof(LC) == CONNECTION_RECORD_SIZE,
              "LC shared-memory record size mismatch");
// Maps one shared-memory segment for the native thought process.
void *map_shm(const char *env_var, size_t size) {
  const char *name = getenv(env_var);
  if (!name) {
    std::cerr << "[THOUGHT_CPP] ERROR: Env var " << env_var << " not set." << std::endl;
    return nullptr;
  }
  std::string shm_name = name;
  if (!shm_name.empty() && shm_name.front() != '/') {
    shm_name.insert(shm_name.begin(), '/');
  }
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd == -1) {
    std::cerr << "[THOUGHT_CPP] ERROR: shm_open failed for " << shm_name
              << " (from " << env_var << ") | Error: " << strerror(errno)
              << std::endl;
    return nullptr;
  }
  void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "[THOUGHT_CPP] ERROR: mmap failed for " << shm_name
              << " | Error: " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }
  close(fd);
  return ptr;
}
// Returns whether an agent index fits the shared-memory bounds.
bool valid_agent_index(int idx) { return idx >= 0 && idx < MAX_AGENTS; }
// Returns Euclidean distance between two shared-memory agent positions.
float distance_between(const AgentPos *positions, int left, int right) {
  const AgentPos &a = positions[left];
  const AgentPos &b = positions[right];
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  const float dist = std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
  return std::max(dist, 0.001f);
}
// Reads route seeds and max-hop settings from the Python-written input file.
bool read_input(const std::string &path, int &max_hops, std::vector<Seed> &seeds) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "[THOUGHT_CPP] ERROR: could not read input file " << path
              << std::endl;
    return false;
  }
  std::string kind;
  while (in >> kind) {
    if (kind == "max_hops") {
      in >> max_hops;
    } else if (kind == "seed") {
      Seed seed{};
      seed.goalIndex = -1;
      seed.routeIndex = 0;
      in >> seed.agentIndex >> seed.spawnCount >> seed.goalIndex;
      std::string rest;
      std::getline(in, rest);
      std::istringstream rest_in(rest);
      rest_in >> seed.routeIndex;
      if (valid_agent_index(seed.agentIndex) && seed.spawnCount > 0 &&
          valid_agent_index(seed.goalIndex)) {
        seeds.push_back(seed);
      }
    } else {
      std::string rest;
      std::getline(in, rest);
    }
  }
  max_hops = std::max(1, max_hops);
  return !seeds.empty();
}
// Chooses the next route edge, favoring reachable unvisited nodes near the current node.
const Edge *choose_edge(const std::vector<Edge> &edges, const AgentPos *positions,
                        int previous, const std::unordered_set<int> &visited,
                        const std::vector<unsigned char> &can_reach_goal,
                        int goal, std::mt19937 &rng) {
  std::vector<const Edge *> candidates;
  std::vector<double> weights;
  candidates.reserve(edges.size());
  weights.reserve(edges.size());
  for (int pass = 0; pass < 3; pass++) {
    candidates.clear();
    weights.clear();
    for (const Edge &edge : edges) {
      if (edge.to == edge.from) {
        continue;
      }
      if (!can_reach_goal.empty() &&
          (!valid_agent_index(edge.to) || !can_reach_goal[edge.to])) {
        continue;
      }
      if (pass == 0 &&
          (edge.to == previous || visited.find(edge.to) != visited.end())) {
        continue;
      }
      if (pass == 1 && edge.to == previous) {
        continue;
      }
      if (edge.to == goal) {
        return &edge;
      }
      const float dist = distance_between(positions, edge.from, edge.to);
      double weight = 1.0 / static_cast<double>(dist);
      candidates.push_back(&edge);
      weights.push_back(weight);
    }
    if (!candidates.empty()) {
      std::discrete_distribution<size_t> distribution(weights.begin(),
                                                       weights.end());
      return candidates[distribution(rng)];
    }
  }
  return nullptr;
}
// Builds reverse reachability and next-hop tables for a goal agent.
GoalRouteTree build_goal_route_tree(
    int goal, const std::vector<std::vector<ReverseEdge>> &reverse_adjacency) {
  GoalRouteTree tree{};
  tree.reachable.assign(MAX_AGENTS, 0);
  tree.nextNode.assign(MAX_AGENTS, -1);
  tree.nextConnection.assign(MAX_AGENTS, -1);
  tree.nextReversed.assign(MAX_AGENTS, 0);
  if (!valid_agent_index(goal)) {
    return tree;
  }
  std::vector<int> frontier;
  frontier.reserve(1024);
  frontier.push_back(goal);
  tree.reachable[goal] = 1;
  tree.nextNode[goal] = goal;
  for (size_t cursor = 0; cursor < frontier.size(); cursor++) {
    int node = frontier[cursor];
    for (const ReverseEdge &edge : reverse_adjacency[node]) {
      if (!valid_agent_index(edge.from) || tree.reachable[edge.from]) {
        continue;
      }
      tree.reachable[edge.from] = 1;
      tree.nextNode[edge.from] = edge.to;
      tree.nextConnection[edge.from] = edge.connectionIndex;
      tree.nextReversed[edge.from] = edge.reversed;
      frontier.push_back(edge.from);
    }
  }
  return tree;
}
// Appends the deterministic route tail from a current node to its goal.
bool append_goal_route_tail(int start, int goal, const GoalRouteTree &tree,
                            std::vector<Step> &path) {
  if (!valid_agent_index(start) || !valid_agent_index(goal)) {
    return false;
  }
  if (start == goal) {
    return true;
  }
  if (tree.reachable.empty() || !tree.reachable[start]) {
    return false;
  }
  int current = start;
  for (int guard = 0; guard < MAX_AGENTS && current != goal; guard++) {
    int next = tree.nextNode[current];
    int connection = tree.nextConnection[current];
    if (!valid_agent_index(next) || connection < 0 || next == current) {
      return false;
    }
    path.push_back(Step{connection, current, next, tree.nextReversed[current]});
    current = next;
  }
  return current == goal;
}
// Runs native route sampling and writes thought paths back for Python to load.
int main() {
  const char *input_env = getenv("THOUGHT_INPUT");
  const char *output_env = getenv("THOUGHT_OUTPUT");
  if (!input_env || !output_env) {
    std::cerr << "[THOUGHT_CPP] ERROR: THOUGHT_INPUT and THOUGHT_OUTPUT are required."
              << std::endl;
    return 1;
  }
  AgentPos *positions =
      (AgentPos *)map_shm("SHM_POS", MAX_AGENTS * sizeof(AgentPos));
  void *connection_base =
      map_shm("SHM_CONNECTIONS", 4 + MAX_CONNECTIONS * CONNECTION_RECORD_SIZE);
  if (!positions || !connection_base) {
    return 1;
  }
  int max_hops = DEFAULT_MAX_HOPS;
  std::vector<Seed> seeds;
  if (!read_input(input_env, max_hops, seeds)) {
    std::cerr << "[THOUGHT_CPP] No usable thought seeds with assigned goals." << std::endl;
    return 2;
  }
  int *connection_count_ptr = (int *)connection_base;
  LC *connections = (LC *)((char *)connection_base + 4);
  int connection_count = *connection_count_ptr;
  connection_count = std::max(0, std::min(connection_count, MAX_CONNECTIONS));
  std::vector<std::vector<Edge>> adjacency(MAX_AGENTS);
  std::vector<std::vector<ReverseEdge>> reverse_adjacency(MAX_AGENTS);
  for (int i = 0; i < connection_count; i++) {
    const LC &connection = connections[i];
    if (!valid_agent_index(connection.idxA) ||
        !valid_agent_index(connection.idxB) || connection.idxA == connection.idxB) {
      continue;
    }
    adjacency[connection.idxA].push_back(
        Edge{i, connection.idxA, connection.idxB, 0});
    reverse_adjacency[connection.idxB].push_back(
        ReverseEdge{i, connection.idxA, connection.idxB, 0});
  }
  std::ofstream out(output_env);
  if (!out) {
    std::cerr << "[THOUGHT_CPP] ERROR: could not write output file "
              << output_env << std::endl;
    return 1;
  }
  std::mt19937 rng(
      static_cast<unsigned int>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  int thought_count = 0;
  int success_count = 0;
  std::unordered_map<int, GoalRouteTree> route_tree_by_goal;
  for (const Seed &seed : seeds) {
    auto route_entry = route_tree_by_goal.find(seed.goalIndex);
    if (route_entry == route_tree_by_goal.end()) {
      route_entry = route_tree_by_goal
                        .emplace(seed.goalIndex,
                                 build_goal_route_tree(seed.goalIndex,
                                                       reverse_adjacency))
                        .first;
    }
    const GoalRouteTree &route_tree = route_entry->second;
    const std::vector<unsigned char> &can_reach_goal = route_tree.reachable;
    for (int spawn = 0; spawn < seed.spawnCount; spawn++) {
      int current = seed.agentIndex;
      int previous = -1;
      std::vector<Step> path;
      std::unordered_set<int> visited;
      visited.insert(current);
      std::string reason = "max_hops";
      bool success = false;
      if (!valid_agent_index(current) || !can_reach_goal[current]) {
        reason = "dead";
      } else {
        for (int hop = 0; hop < max_hops; hop++) {
          if (current == seed.goalIndex) {
            reason = "endpoint";
            success = true;
            break;
          }
          if (!valid_agent_index(current) || adjacency[current].empty()) {
            break;
          }
          const Edge *edge = choose_edge(adjacency[current], positions, previous,
                                         visited, can_reach_goal, seed.goalIndex,
                                         rng);
          if (!edge) {
            break;
          }
          path.push_back(
              Step{edge->connectionIndex, edge->from, edge->to, edge->reversed});
          previous = current;
          current = edge->to;
          visited.insert(current);
        }
      }
      if (!success) {
        if (current == seed.goalIndex ||
            append_goal_route_tail(current, seed.goalIndex, route_tree, path)) {
          current = seed.goalIndex;
          reason = "endpoint";
          success = true;
        }
      }
      if (success) {
        success_count++;
      }
      thought_count++;
      out << "thought " << seed.agentIndex << " " << current << " " << reason
          << " " << (success ? 1 : 0) << " " << path.size() << " "
          << seed.goalIndex << " " << seed.routeIndex << "\n";
      for (const Step &step : path) {
        out << "step " << step.connectionIndex << " " << step.from << " "
            << step.to << " " << step.reversed << "\n";
      }
      out << "end\n";
    }
  }
  std::cout << "[THOUGHT_CPP] Completed " << thought_count << " thoughts with "
            << success_count << " assigned endpoints." << std::endl;
  return 0;
}
