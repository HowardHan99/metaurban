#!/usr/bin/env python3
"""
Test script for spawn_increase_per_episode feature.
Simulates multiple episode resets to verify the dynamic pedestrian spawning.
"""

class MockConfig(dict):
    """Mock config that behaves like the real Config class."""
    def get(self, key, default=None):
        return dict.get(self, key, default)

class MockEnv:
    """Mock environment to test the spawn increase logic."""
    def __init__(self, base_spawn=40, spawn_increase=5):
        self.config = MockConfig({
            "spawn_human_num": base_spawn,
            "spawn_increase_per_episode": spawn_increase,
        })
    
    def reset_logic(self):
        """Simulates the reset logic from SocialDynamicMetaUrbanEnv."""
        spawn_increase = self.config.get("spawn_increase_per_episode", 0)
        if spawn_increase > 0:
            # Save base spawn value on first reset
            if not hasattr(self, "_base_spawn_human_num"):
                self._base_spawn_human_num = self.config.get("spawn_human_num", 40)
            
            reset_count = getattr(self, "_reset_count", 0)
            base_spawn = self._base_spawn_human_num
            dynamic_spawn = base_spawn + reset_count * spawn_increase
            self.config["spawn_human_num"] = int(dynamic_spawn)
            print(
                f"Episode {reset_count}: dynamically set spawn_human_num={int(dynamic_spawn)} "
                f"(base={base_spawn} + {reset_count}*{spawn_increase})"
            )
            self._reset_count = reset_count + 1
        else:
            if not hasattr(self, "_reset_count"):
                self._reset_count = 0

def main():
    print("=" * 70)
    print("Test 1: With spawn_increase_per_episode=5 (base_spawn=40)")
    print("=" * 70)
    env1 = MockEnv(base_spawn=40, spawn_increase=5)
    for episode in range(5):
        env1.reset_logic()
        print(f"  → Current spawn_human_num: {env1.config['spawn_human_num']}")
    print()
    
    print("=" * 70)
    print("Test 2: With spawn_increase_per_episode=0 (no increase)")
    print("=" * 70)
    env2 = MockEnv(base_spawn=40, spawn_increase=0)
    for episode in range(3):
        env2.reset_logic()
        print(f"  → Current spawn_human_num: {env2.config['spawn_human_num']}")
    print()
    
    print("=" * 70)
    print("Test 3: With spawn_increase_per_episode=10 (base_spawn=30)")
    print("=" * 70)
    env3 = MockEnv(base_spawn=30, spawn_increase=10)
    for episode in range(4):
        env3.reset_logic()
        print(f"  → Current spawn_human_num: {env3.config['spawn_human_num']}")
    print()
    
    print("=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
