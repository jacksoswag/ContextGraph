#include "raylib.h"
#include "raymath.h"
#include <cmath>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

const int MAX_AGENTS = 60000;
const int MAX_CONNECTIONS = 80000;
const float DAMPING = 0.99f;
const float ATTRACT_FORCE = 3.5f;
const float REPULSION_FORCE = 450.0f;
const float BOUNDS = 1000.0f;

// STABILITY PARAMETERS
const float STABILITY_THRESHOLD = 0.001f; // Energy threshold for "stable"
const int STABILITY_REQUIRED_FRAMES =
    120; // Must be stable for 2 seconds (at 60fps)

// PHASE CONSTANTS (Must match constants.py)
const int PHASE_PHYSICS = 7;
const int PHASE_STABLE = 8;

struct AgentPos {
  float x, y, z;
  float vx, vy, vz;
};

struct LC {
  int idxA;
  int type;
  int flags;
  int idxB;
};

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
      (AgentPos *)map_shm("SHM_SKS_POS", MAX_AGENTS * sizeof(AgentPos));
  void *LC_base = map_shm("SHM_SKS_CONNECTIONS", 4 + MAX_CONNECTIONS * 16);
  unsigned char *status_buf = (unsigned char *)map_shm("SHM_SKS_STATUS", 1024);

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
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> initial_velocity(-1.0f, 1.0f);

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
    // Only run physics if the system is in PHASE_PHYSICS (7)
    if (status_buf[0] != PHASE_PHYSICS) {
      if (status_buf[0] == PHASE_STABLE)
        break;        // We are already done
      usleep(100000); // 10Hz poll while waiting
      continue;
    }
    int current_LCs = *LC_count;
    if (current_LCs > MAX_CONNECTIONS)
      current_LCs = MAX_CONNECTIONS;

    float total_kinetic_energy = 0.0f;
    int active_agents = 0;

    const int LOGICAL_CONNECTOR_COUNT =
        3; // Seeded logical connectors: 0 = be, 1 = lead to, 2 = subset of.

    // 1. SPRING PHYSICS
    for (int i = 0; i < current_LCs; i++) {
      LC &b = LC_buf[i];
      if (b.type >= LOGICAL_CONNECTOR_COUNT)
        continue;
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

      bool isTrue = ((b.flags & 1) != 0);
      float force = isTrue ? (dist * ATTRACT_FORCE) : (-REPULSION_FORCE / dist);

      a.vx += dir.x * force * 0.008f;
      a.vy += dir.y * force * 0.008f;
      a.vz += dir.z * force * 0.008f;
    }

    // 2. INTEGRATION & ENERGY CALC
    for (int i = 0; i < MAX_AGENTS; i++) {
      AgentPos &p = pos_buf[i];
      if (p.x == 0 && p.y == 0 && p.z == 0 && p.vx == 0 && p.vy == 0 &&
          p.vz == 0)
        continue;

      active_agents++;

      p.x += p.vx;
      p.y += p.vy;
      p.z += p.vz;

      p.vx *= DAMPING;
      p.vy *= DAMPING;
      p.vz *= DAMPING;

      // Calculate Kinetic Energy: v^2
      total_kinetic_energy += (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);

      // Bounds
      float distSq = p.x * p.x + p.y * p.y + p.z * p.z;
      if (distSq > (BOUNDS * BOUNDS)) {
        float d = sqrtf(distSq);
        float scale = BOUNDS / d;
        p.x *= scale;
        p.y *= scale;
        p.z *= scale;
        p.vx *= -0.5f;
        p.vy *= -0.5f;
        p.vz *= -0.5f;
      }
    }

    // 3. STABILITY CHECK
    float avg_energy =
        (active_agents > 0) ? (total_kinetic_energy / active_agents) : 0.0f;

    if (frame_count >
        100) { // Allow initial "explosion" before checking stability
      if (avg_energy < STABILITY_THRESHOLD) {
        stable_frames++;
      } else {
        stable_frames = 0;
      }
    }

    if (frame_count % 60 == 0) {
      std::cout << "[PHYSICS] Avg Energy: " << avg_energy
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

  return 0;
}
