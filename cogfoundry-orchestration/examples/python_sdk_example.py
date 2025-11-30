"""
🔧 CogFoundry Python SDK Example
Demonstrates integration with the Universal Kernel Generator and Orchestration Engine
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import with correct module names (files have hyphens, need to import differently)
import importlib.util

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules
ukg = import_from_path("universal_kernel_generator", parent_dir / "universal_kernel_generator.py")
onto = import_from_path("ontogenesis", parent_dir / "ontogenesis.py")
orch = import_from_path("orchestration_engine", parent_dir / "orchestration-engine.py")

UniversalKernelGenerator = ukg.UniversalKernelGenerator
DomainType = ukg.DomainType
EvolutionEngine = onto.EvolutionEngine
LivingKernel = onto.LivingKernel
CogFoundryOrchestrationEngine = orch.CogFoundryOrchestrationEngine
CogFoundryConfig = orch.CogFoundryConfig


async def example_kernel_generation():
    """Example: Generate domain-specific computational kernels"""
    
    print("=" * 70)
    print("Example 1: Generating Domain-Specific Kernels")
    print("=" * 70)
    
    generator = UniversalKernelGenerator()
    
    # Generate a physics kernel
    print("\n🔬 Generating Physics Kernel (Hamiltonian mechanics)...")
    physics_kernel = generator.generate_physics_kernel(order=4)
    
    print(f"  ✓ Generated kernel with {len(physics_kernel.elementary_differentials)} differentials")
    print(f"  ✓ Grip metric: {physics_kernel.grip_metric:.4f}")
    print(f"  ✓ Stability: {physics_kernel.stability_score:.4f}")
    
    # Generate a consciousness kernel
    print("\n🧠 Generating Consciousness Kernel (echo state networks)...")
    consciousness_kernel = generator.generate_consciousness_kernel(order=4)
    
    print(f"  ✓ Generated kernel with {len(consciousness_kernel.elementary_differentials)} differentials")
    print(f"  ✓ Grip metric: {consciousness_kernel.grip_metric:.4f}")
    print(f"  ✓ Efficiency: {consciousness_kernel.efficiency_score:.4f}")


async def example_kernel_evolution():
    """Example: Evolve self-optimizing kernels"""
    
    print("\n" + "=" * 70)
    print("Example 2: Evolving Self-Optimizing Kernels")
    print("=" * 70)
    
    # Create evolution engine
    print("\n🌱 Creating Evolution Engine...")
    engine = EvolutionEngine(
        population_size=15,
        domain=DomainType.COMPUTING,
        kernel_order=4
    )
    
    # Initialize population
    print("  Initializing population...")
    engine.initialize_population()
    
    initial_fitness = engine.get_best_kernel().fitness
    print(f"  ✓ Initial best fitness: {initial_fitness:.4f}")
    
    # Run evolution
    print("\n🧬 Running evolution (20 generations)...")
    history = engine.run_evolution(max_generations=20, fitness_threshold=0.95)
    
    # Get best kernel
    best_kernel = engine.get_best_kernel()
    
    print(f"\n✨ Evolution Complete!")
    print(f"  Final best fitness: {best_kernel.fitness:.4f}")
    print(f"  Improvement: {best_kernel.fitness - initial_fitness:.4f}")
    print(f"  Generation: {best_kernel.genome.generation}")
    print(f"  Development stage: {best_kernel.development_stage.value}")


async def example_orchestration():
    """Example: Orchestrate deployment across cognitive cities"""
    
    print("\n" + "=" * 70)
    print("Example 3: Orchestrating Cognitive Cities Deployment")
    print("=" * 70)
    
    # Create orchestrator
    print("\n🏗️ Creating CogFoundry Orchestrator...")
    config = CogFoundryConfig()
    orchestrator = CogFoundryOrchestrationEngine(config)
    
    # Initialize
    print("  Initializing...")
    if await orchestrator.initialize():
        print("  ✓ Orchestrator initialized successfully")
        
        # Deploy neural architecture
        print("\n🚀 Deploying Neural Architecture...")
        deployment_spec = {
            "name": "example_deployment",
            "architecture_type": "distributed_transformer",
            "target_cities": [
                "github.com/organizations/cogcities",
                "github.com/organizations/cogpilot"
            ]
        }
        
        result = await orchestrator.orchestrate_deployment(deployment_spec)
        print(f"  ✓ Deployment ID: {result['deployment_id']}")
        print(f"  ✓ Status: {result['coordination_status']}")
        
        # Deploy evolved kernel
        print("\n🧬 Deploying Evolved Computing Kernel...")
        kernel_result = await orchestrator.orchestrate_kernel_deployment(
            domain="computing",
            target_cities=[
                "github.com/organizations/cogcities",
                "github.com/organizations/cogpilot"
            ]
        )
        
        print(f"  ✓ Domain: {kernel_result['domain']}")
        print(f"  ✓ Best fitness: {kernel_result['best_fitness']:.4f}")
        print(f"  ✓ Kernel generation: {kernel_result['kernel_generation']}")
        
        # Monitor health
        print("\n📊 Monitoring Ecosystem Health...")
        health = await orchestrator.monitor_ecosystem_health()
        print(f"  ✓ Connected cities: {health['connected_cities']}")
        print(f"  ✓ Active deployments: {health['active_deployments']}")
        print(f"  ✓ Autognosis progression: {health['autognosis_progression']:.2f}")
        
    else:
        print("  ✗ Failed to initialize orchestrator")


async def main():
    """Run all examples"""
    
    print("\n🔬 CogFoundry Python SDK Examples")
    print("=" * 70)
    print("Demonstrating Universal Kernel Generation, Ontogenesis, and Orchestration")
    print()
    
    # Run examples
    await example_kernel_generation()
    await example_kernel_evolution()
    await example_orchestration()
    
    print("\n" + "=" * 70)
    print("✨ All examples completed successfully!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
