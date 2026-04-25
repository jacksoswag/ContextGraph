#include "raylib.h"
#include "raymath.h"
#include <chrono>
#include <cmath>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

const int MAX_AGENTS = 64000;
const int MAX_CONNECTIONS = 80000;
const int CONNECTION_RECORD_SIZE = 40;
const float DAMPING = 0.94f;
const float REST_DISTANCE = 32.0f;
const float REPEL_EQUIL_DIST = 128.0f;
const float MIN_SPRING_STRENGTH = 0.014f;
const float MAX_SPRING_STRENGTH = 0.075f;
const float MAX_SPRING_FORCE = 3.25f;
const float FORCE_TIMESTEP = 0.035f;
const float BASE_BOUNDS = 520.0f;
const float BOUNDS_PER_SQRT_AGENT = 30.0f;
const float SOFT_BOUND_FORCE = 0.0018f;
const float REPULSION_RADIUS = 20.0f;
const float REPULSION_STRENGTH = 4.85f;
const float MAX_REPULSION_FORCE = 6.4f;
const int MAX_SIMULATION_SECONDS = 22;

// STABILITY PARAMETERS
const float STABILITY_THRESHOLD = 0.015f; // Energy threshold for "stable"
const float STABILITY_AVG_STRETCH_THRESHOLD = 0.65f;
const float STABILITY_MAX_STRETCH_THRESHOLD = 2.25f;
const int STABILITY_REQUIRED_FRAMES =
    90; // Must be stable for 1.5 seconds (at 60fps)

// PHASE CONSTANTS (Must match constants.py)
const int PHASE_PHYSICS = 2;
const int PHASE_STABLE = 3;

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

bool is_negative_connection(const LC &connection) {
  return connection.subjectTruthExact >= 0 ||
         connection.predicateTruthExact >= 0;
}

struct GridKey {
  int x, y, z;

  bool operator==(const GridKey &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct GridKeyHash {
  size_t operator()(const GridKey &key) const {
    size_t hx = static_cast<size_t>(key.x) * 73856093u;
    size_t hy = static_cast<size_t>(key.y) * 19349663u;
    size_t hz = static_cast<size_t>(key.z) * 83492791u;
    return hx ^ hy ^ hz;
  }
};

static_assert(sizeof(LC) == CONNECTION_RECORD_SIZE,
              "LC shared-memory record size mismatch");

void *map_shm(const char *env_var, size_t size) {
  const char *name = getenv(env_var);
  if (!name) {
    std::cerr << "[PHYSICS] ERROR: Env var " << env_var << " not set."
              << std::endl;
    return nullptr;
  }
  std::string shm_name = name;
  if (!shm_name.empty() && shm_name.front() != '/') {
    shm_name.insert(shm_name.begin(), '/');
  }
  int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (fd == -1) {
    std::cerr << "[PHYSICS] ERROR: shm_open failed for " << shm_name
              << " (from " << env_var << ") | Error: " << strerror(errno)
              << std::endl;
    return nullptr;
  }
  void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    std::cerr << "[PHYSICS] ERROR: mmap failed for " << shm_name
              << " | Error: " << strerror(errno) << std::endl;
    close(fd);
    return nullptr;
  }
  close(fd);
  return ptr;
}

int main() {
  std::cout << "[PHYSICS] Phased Simulation Engine Started." << std::endl;

  AgentPos *pos_buf =
      (AgentPos *)map_shm("SHM_POS", MAX_AGENTS * sizeof(AgentPos));
  void *LC_base =
      map_shm("SHM_CONNECTIONS", 4 + MAX_CONNECTIONS * CONNECTION_RECORD_SIZE);
  unsigned char *status_buf = (unsigned char *)map_shm("SHM_STATUS", 1024);

  if (!pos_buf || !LC_base || !status_buf) {
    std::cerr << "[PHYSICS] FATAL: Could not map shared memory. Check if "
                 "segments were created."
              << std::endl;
    return 1;
  }

  int *LC_count = (int *)LC_base;
  LC *LC_buf = (LC *)((char *)LC_base + 4);

  int stable_frames = 0;
  long frame_count = 0;
  bool physics_started = false;
  bool timed_out = false;
  std::chrono::steady_clock::time_point physics_start_time;
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> initial_velocity(-0.035f, 0.035f);
  std::unordered_map<GridKey, std::vector<int>, GridKeyHash> repulsion_grid;
  std::vector<int> active_indices;

  // Give every initialized agent a small random starting push so the graph can
  // unfold without waking up unused zeroed slots in shared memory.
  for (int i = 0; i < MAX_AGENTS; i++) {
    AgentPos &p = pos_buf[i];
    if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f)
      continue;
    p.vx = initial_velocity(rng);
    p.vy = initial_velocity(rng);
    p.vz = initial_velocity(rng);
  }

  while (true) {
    // Only run physics if the system is in PHASE_PHYSICS.
    if (status_buf[0] != PHASE_PHYSICS) {
      if (status_buf[0] == PHASE_STABLE)
        break;        // We are already done
      usleep(100000); // 10Hz poll while waiting
      continue;
    }

    if (!physics_started) {
      physics_started = true;
      physics_start_time = std::chrono::steady_clock::now();
    }

    const auto elapsed = std::chrono::steady_clock::now() - physics_start_time;
    if (elapsed >= std::chrono::seconds(MAX_SIMULATION_SECONDS)) {
      std::cout << "[PHYSICS] Reached " << MAX_SIMULATION_SECONDS
                << "s simulation limit. Exiting phase without stabilization."
                << std::endl;
      timed_out = true;
      break;
    }

    int current_LCs = *LC_count;
    if (current_LCs > MAX_CONNECTIONS)
      current_LCs = MAX_CONNECTIONS;

    float total_kinetic_energy = 0.0f;
    float total_spring_stretch = 0.0f;
    float max_spring_stretch = 0.0f;
    int active_springs = 0;
    int active_agents = 0;
    Vector3 swarm_center = {0.0f, 0.0f, 0.0f};
    active_indices.clear();

    for (int i = 0; i < MAX_AGENTS; i++) {
      AgentPos &p = pos_buf[i];
      if (p.x == 0 && p.y == 0 && p.z == 0 && p.vx == 0 && p.vy == 0 &&
          p.vz == 0)
        continue;
      active_indices.push_back(i);
      active_agents++;
      swarm_center.x += p.x;
      swarm_center.y += p.y;
      swarm_center.z += p.z;
    }

    if (active_agents > 0) {
      float inv_agents = 1.0f / static_cast<float>(active_agents);
      swarm_center.x *= inv_agents;
      swarm_center.y *= inv_agents;
      swarm_center.z *= inv_agents;
    }

    float dynamic_bounds =
        BASE_BOUNDS +
        (std::sqrt(static_cast<float>(std::max(active_agents, 1))) *
         BOUNDS_PER_SQRT_AGENT);

    // 1. CONNECTION PHYSICS
    for (int i = 0; i < current_LCs; i++) {
      LC &b = LC_buf[i];
      if (b.idxA < 0 || b.idxA >= MAX_AGENTS || b.idxB < 0 ||
          b.idxB >= MAX_AGENTS)
        continue;

      AgentPos &a = pos_buf[b.idxA];
      AgentPos &b_pos = pos_buf[b.idxB];

      Vector3 delta = {b_pos.x - a.x, b_pos.y - a.y, b_pos.z - a.z};
      float dist = Vector3Length(delta);
      if (dist < 1.0f)
        dist = 1.0f;
      Vector3 dir = Vector3Scale(delta, 1.0f / dist);

      float utility = b.utility;
      if (utility < 0.0f)
        utility = 0.0f;
      if (utility > 1.0f)
        utility = 1.0f;

      float equilibrium_distance =
          is_negative_connection(b) ? REPEL_EQUIL_DIST : REST_DISTANCE;
      float displacement = dist - equilibrium_distance;
      float normalized_stretch = std::fabs(displacement) / equilibrium_distance;
      total_spring_stretch += normalized_stretch;
      if (normalized_stretch > max_spring_stretch)
        max_spring_stretch = normalized_stretch;
      active_springs++;

      float spring_strength =
          MIN_SPRING_STRENGTH +
          (utility * utility * (MAX_SPRING_STRENGTH - MIN_SPRING_STRENGTH));
      float force = displacement * spring_strength;
      if (force > MAX_SPRING_FORCE)
        force = MAX_SPRING_FORCE;
      if (force < -MAX_SPRING_FORCE)
        force = -MAX_SPRING_FORCE;
      force *= FORCE_TIMESTEP;
      a.vx += dir.x * force;
      a.vy += dir.y * force;
      a.vz += dir.z * force;
      b_pos.vx -= dir.x * force;
      b_pos.vy -= dir.y * force;
      b_pos.vz -= dir.z * force;
    }

    // 2. LOCAL NODE REPULSION
    repulsion_grid.clear();
    repulsion_grid.reserve(active_indices.size() * 2);
    for (int idx : active_indices) {
      AgentPos &p = pos_buf[idx];
      GridKey key = {
          static_cast<int>(std::floor(p.x / REPULSION_RADIUS)),
          static_cast<int>(std::floor(p.y / REPULSION_RADIUS)),
          static_cast<int>(std::floor(p.z / REPULSION_RADIUS)),
      };
      repulsion_grid[key].push_back(idx);
    }

    for (int idx : active_indices) {
      AgentPos &a = pos_buf[idx];
      GridKey center_key = {
          static_cast<int>(std::floor(a.x / REPULSION_RADIUS)),
          static_cast<int>(std::floor(a.y / REPULSION_RADIUS)),
          static_cast<int>(std::floor(a.z / REPULSION_RADIUS)),
      };

      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          for (int dz = -1; dz <= 1; dz++) {
            GridKey neighbor_key = {
                center_key.x + dx,
                center_key.y + dy,
                center_key.z + dz,
            };
            auto it = repulsion_grid.find(neighbor_key);
            if (it == repulsion_grid.end())
              continue;

            for (int other_idx : it->second) {
              if (other_idx <= idx)
                continue;

              AgentPos &b = pos_buf[other_idx];
              Vector3 delta = {b.x - a.x, b.y - a.y, b.z - a.z};
              float dist = Vector3Length(delta);
              if (dist >= REPULSION_RADIUS)
                continue;

              Vector3 dir;
              if (dist < 0.001f) {
                float angle = static_cast<float>((idx % 997) + 1) * 0.0174533f;
                dir = {std::cos(angle), std::sin(angle),
                       std::cos(angle * 0.5f)};
                dir = Vector3Normalize(dir);
                dist = 0.001f;
              } else {
                dir = Vector3Scale(delta, 1.0f / dist);
              }

              float proximity = 1.0f - (dist / REPULSION_RADIUS);
              float force = proximity * proximity * REPULSION_STRENGTH;
              if (force > MAX_REPULSION_FORCE)
                force = MAX_REPULSION_FORCE;
              force *= FORCE_TIMESTEP;

              a.vx -= dir.x * force;
              a.vy -= dir.y * force;
              a.vz -= dir.z * force;
              b.vx += dir.x * force;
              b.vy += dir.y * force;
              b.vz += dir.z * force;
            }
          }
        }
      }
    }

    // 3. INTEGRATION & ENERGY CALC
    total_kinetic_energy = 0.0f;
    for (int idx : active_indices) {
      AgentPos &p = pos_buf[idx];

      p.x += p.vx;
      p.y += p.vy;
      p.z += p.vz;

      p.vx *= DAMPING;
      p.vy *= DAMPING;
      p.vz *= DAMPING;

      // Calculate Kinetic Energy: v^2
      total_kinetic_energy += (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);

      // Soft dynamic bounds around the swarm centroid.
      Vector3 offset = {
          p.x - swarm_center.x,
          p.y - swarm_center.y,
          p.z - swarm_center.z,
      };
      float radial_distance = Vector3Length(offset);
      if (radial_distance > dynamic_bounds) {
        float overflow = radial_distance - dynamic_bounds;
        Vector3 radial_dir =
            Vector3Scale(offset, 1.0f / std::max(radial_distance, 1.0f));
        float pull = overflow * SOFT_BOUND_FORCE;
        p.vx -= radial_dir.x * pull;
        p.vy -= radial_dir.y * pull;
        p.vz -= radial_dir.z * pull;
      }
    }

    // 3. STABILITY CHECK
    float avg_energy =
        (active_agents > 0) ? (total_kinetic_energy / active_agents) : 0.0f;
    float avg_spring_stretch =
        (active_springs > 0) ? (total_spring_stretch / active_springs) : 0.0f;
    bool layout_resolved =
        active_springs == 0 ||
        (avg_spring_stretch < STABILITY_AVG_STRETCH_THRESHOLD &&
         max_spring_stretch < STABILITY_MAX_STRETCH_THRESHOLD);

    if (frame_count >
        100) { // Allow initial "explosion" before checking stability
      if (avg_energy < STABILITY_THRESHOLD && layout_resolved) {
        stable_frames++;
      } else {
        stable_frames = 0;
      }
    }

    if (frame_count % 60 == 0) {
      std::cout << "[PHYSICS] Avg Energy: " << avg_energy
                << " | Avg Stretch: " << avg_spring_stretch
                << " | Max Stretch: " << max_spring_stretch
                << " | Springs: " << active_springs
                << " | Stable: " << stable_frames << "/"
                << STABILITY_REQUIRED_FRAMES << std::endl;
    }

    if (stable_frames >= STABILITY_REQUIRED_FRAMES) {
      std::cout << "[PHYSICS] SWARM STABILIZED. Exiting phase." << std::endl;
      break;
    }

    frame_count++;
    usleep(16000); // ~60 FPS
  }

  return timed_out ? 2 : 0;
}
