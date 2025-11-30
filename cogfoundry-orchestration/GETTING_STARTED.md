# 🚀 CogFoundry Getting Started Guide

## Welcome to CogFoundry!

CogFoundry transforms Foundry Local into a distributed cognitive architecture orchestration engine, enabling deployment of self-evolving AI Neural Architectures across Networks of Cognitive Cities.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Examples](#examples)
7. [API Reference](#api-reference)

## Quick Start

### 1. Generate Your First Kernel

```python
from universal_kernel_generator import UniversalKernelGenerator, DomainType

# Create generator
generator = UniversalKernelGenerator()

# Generate a computing kernel
kernel = generator.generate_computing_kernel(order=4)

print(f"Grip Metric: {kernel.grip_metric:.4f}")
print(f"Stability: {kernel.stability_score:.4f}")
```

### 2. Evolve Self-Optimizing Kernels

```python
from ontogenesis import EvolutionEngine, DomainType

# Create evolution engine
engine = EvolutionEngine(
    population_size=20,
    domain=DomainType.COMPUTING,
    kernel_order=4
)

# Initialize and evolve
engine.initialize_population()
history = engine.run_evolution(max_generations=30)

# Get best kernel
best_kernel = engine.get_best_kernel()
print(f"Best Fitness: {best_kernel.fitness:.4f}")
```

### 3. Orchestrate Cognitive Cities

```python
from orchestration_engine import CogFoundryOrchestrationEngine, CogFoundryConfig
import asyncio

async def deploy():
    # Create orchestrator
    orchestrator = CogFoundryOrchestrationEngine()
    
    # Initialize
    await orchestrator.initialize()
    
    # Deploy kernel
    result = await orchestrator.orchestrate_kernel_deployment(
        domain="computing",
        target_cities=["github.com/organizations/cogcities"]
    )
    
    print(f"Deployment: {result['domain']}")
    print(f"Fitness: {result['best_fitness']:.4f}")

asyncio.run(deploy())
```

## Core Concepts

### 🔬 Universal Kernel Generator

Generates domain-specific computational kernels through differential calculus using **B-series expansion** following OEIS A000081 (rooted tree enumeration).

**Key Features:**
- Elementary differential computation
- Grip metric optimization
- Domain-specific kernels (Physics, Consciousness, Computing, Biology)

**Example Domains:**
- **Physics**: Hamiltonian mechanics, phase-space composition
- **Consciousness**: Echo trees, memory composition, gestalt formation
- **Computing**: Recursion trees, function composition
- **Biology**: Metabolic trees, cascade composition

### 🌱 Ontogenesis System

Mathematical structures that exhibit life-like properties:

**Life Operations:**
- **Self-Generation**: Recursive self-composition (chain rule)
- **Self-Optimization**: Iterative grip improvement
- **Self-Reproduction**: Crossover and mutation
- **Evolution**: Population-based fitness optimization

**Development Stages:**
- Embryonic → Juvenile → Mature → Senescent

**Fitness Function:**
```
fitness = grip(0.4) + stability(0.2) + efficiency(0.2) + novelty(0.1) + symmetry(0.1)
```

### 🏗️ Orchestration Engine

Principal Architect coordinating the Cognitive Cities Ecosystem:

**Components:**
- **MCP Master Builder**: Model Context Protocol for CogPilot integration
- **Meta-LSP Protocols**: Introspective development extensions
- **VM-Daemon MLOps**: Service architecture and maintenance
- **Neural Transport**: High-bandwidth cognitive city communication

## Installation

### Prerequisites

```bash
# Python 3.11+ required
python3 --version

# Install dependencies
pip install numpy
```

### Setup CogFoundry

```bash
# Clone repository
git clone https://github.com/o9nn/cogfoundry
cd cogfoundry/cogfoundry-orchestration

# Run demonstration
python3 orchestration-engine.py
```

## Basic Usage

### Generating Kernels

```python
from universal_kernel_generator import UniversalKernelGenerator, DomainType

generator = UniversalKernelGenerator()

# Generate physics kernel
physics_kernel = generator.generate_physics_kernel(order=4)
print(f"Physics Grip: {physics_kernel.grip_metric:.4f}")

# Generate consciousness kernel
consciousness_kernel = generator.generate_consciousness_kernel(order=4)
print(f"Consciousness Efficiency: {consciousness_kernel.efficiency_score:.4f}")
```

### Running Evolution

```python
from ontogenesis import EvolutionEngine, DomainType

# Create engine
engine = EvolutionEngine(
    population_size=20,
    domain=DomainType.PHYSICS,
    kernel_order=4
)

# Initialize population
engine.initialize_population()

# Evolve for 50 generations
history = engine.run_evolution(
    max_generations=50,
    fitness_threshold=0.9
)

# Access best kernel
best = engine.get_best_kernel()
print(f"Generation: {best.genome.generation}")
print(f"Fitness: {best.fitness:.4f}")
print(f"Stage: {best.development_stage.value}")
```

### Orchestrating Deployments

```python
import asyncio
from orchestration_engine import CogFoundryOrchestrationEngine

async def main():
    # Initialize orchestrator
    orchestrator = CogFoundryOrchestrationEngine()
    await orchestrator.initialize()
    
    # Deploy neural architecture
    result = await orchestrator.orchestrate_deployment({
        "name": "my_deployment",
        "target_cities": [
            "github.com/organizations/cogcities",
            "github.com/organizations/cogpilot"
        ]
    })
    
    print(f"Deployment ID: {result['deployment_id']}")
    
    # Monitor ecosystem health
    health = await orchestrator.monitor_ecosystem_health()
    print(f"Connected Cities: {health['connected_cities']}")
    print(f"Active Deployments: {health['active_deployments']}")

asyncio.run(main())
```

## Advanced Features

### Custom Domain Kernels

Create kernels for custom domains:

```python
from universal_kernel_generator import UniversalKernelGenerator, ComputationalKernel, DomainType

generator = UniversalKernelGenerator()

# Define custom domain topology
custom_topology = {
    "symmetries": [2, 3, 5],
    "required_orders": [1, 2, 3],
    "invariants": ["custom_invariant"],
    "flow_type": "custom_flow"
}

# Generate differentials
differentials = generator.generate_elementary_differentials(order=4)

# Create custom kernel
kernel = ComputationalKernel(
    domain=DomainType.GENERAL,
    order=4,
    elementary_differentials=differentials,
    b_series_coefficients=np.array([d.coefficient for d in differentials])
)

# Optimize grip
optimized = generator.optimize_grip(kernel, custom_topology, iterations=100)
print(f"Custom Grip: {optimized.grip_metric:.4f}")
```

### Kernel Reproduction

Reproduce kernels with genetic crossover:

```python
from ontogenesis import LivingKernel, KernelGenome

# Assume kernel1 and kernel2 are existing LivingKernel instances
offspring = kernel1.reproduce(kernel2)

print(f"Parent 1 Generation: {kernel1.genome.generation}")
print(f"Parent 2 Generation: {kernel2.genome.generation}")
print(f"Offspring Generation: {offspring.genome.generation}")
print(f"Offspring Mutations: {offspring.genome.mutations}")
```

### Multi-Domain Evolution

Evolve kernels across multiple domains:

```python
from ontogenesis import EvolutionEngine, DomainType

domains = [DomainType.PHYSICS, DomainType.COMPUTING, DomainType.CONSCIOUSNESS]
engines = {}

for domain in domains:
    engine = EvolutionEngine(population_size=15, domain=domain, kernel_order=4)
    engine.initialize_population()
    history = engine.run_evolution(max_generations=30)
    engines[domain] = engine
    
    best = engine.get_best_kernel()
    print(f"{domain.value}: Fitness={best.fitness:.4f}")
```

## Examples

### Complete Python SDK Example

Run the comprehensive example:

```bash
cd cogfoundry-orchestration/examples
python3 python_sdk_example.py
```

This demonstrates:
1. Kernel generation for multiple domains
2. Evolutionary optimization
3. Orchestration across cognitive cities
4. Ecosystem health monitoring

### Individual Component Examples

Each component can be tested independently:

```bash
# Test kernel generator
python3 universal_kernel_generator.py

# Test ontogenesis
python3 ontogenesis.py

# Test orchestration
python3 orchestration-engine.py
```

## API Reference

### UniversalKernelGenerator

**Methods:**
- `generate_elementary_differentials(max_order: int) -> List[ElementaryDifferential]`
- `compute_grip_metric(kernel, domain_topology) -> float`
- `optimize_grip(kernel, domain_topology, iterations=100) -> ComputationalKernel`
- `generate_physics_kernel(order=4) -> ComputationalKernel`
- `generate_consciousness_kernel(order=4) -> ComputationalKernel`
- `generate_computing_kernel(order=4) -> ComputationalKernel`
- `generate_biology_kernel(order=4) -> ComputationalKernel`

### EvolutionEngine

**Methods:**
- `initialize_population(seed_kernels=None)`
- `tournament_selection(tournament_size=3) -> LivingKernel`
- `evolve_generation() -> Dict`
- `run_evolution(max_generations=50, fitness_threshold=0.9) -> List[Dict]`
- `get_best_kernel() -> LivingKernel`

### CogFoundryOrchestrationEngine

**Methods:**
- `async initialize() -> bool`
- `async orchestrate_deployment(deployment_spec: Dict) -> Dict`
- `async orchestrate_kernel_deployment(domain: str, target_cities: List[str]) -> Dict`
- `async monitor_ecosystem_health() -> Dict`
- `save_state(filepath=None)`

## 🎓 Next Steps

1. **Explore the Roadmap**: See `roadmap/agi-emergence-roadmap.md` for the path to Autognosis & Autogenesis
2. **Review Architecture**: Check `../COSMO-ENTERPRISE-TOPOLOGY.md` for enterprise patterns
3. **Join the Community**: Contribute to the cognitive cities ecosystem
4. **Build Custom Protocols**: Create MCP extensions for your domain

## 📚 Additional Resources

- [CogFoundry Orchestration README](./README.md)
- [MCP Master Builder Documentation](./mcp-master-builder/README.md)
- [Meta-LSP Protocols Guide](./meta-lsp-protocols/README.md)
- [AGI Emergence Roadmap](./roadmap/agi-emergence-roadmap.md)

---

**Welcome to the future of distributed cognitive architectures!** 🚀

*"Order from Chaos through Distributed Cognition"*
