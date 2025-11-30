"""
🌱 Ontogenesis: Self-Generating Kernel Evolution
Mathematical structures that exhibit life-like properties

Implements:
- Kernel genomes (coefficient genes + symmetry genes)
- Life operations (self-generation, self-optimization, self-reproduction, evolution)
- Development stages (embryonic → juvenile → mature → senescent)
- Population-based evolutionary optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from copy import deepcopy

from universal_kernel_generator import (
    ComputationalKernel, UniversalKernelGenerator, DomainType
)

logger = logging.getLogger(__name__)


class DevelopmentStage(Enum):
    """Life stages of a kernel"""
    EMBRYONIC = "embryonic"
    JUVENILE = "juvenile"
    MATURE = "mature"
    SENESCENT = "senescent"


@dataclass
class KernelGenome:
    """
    Genetic structure of a computational kernel
    """
    coefficient_genes: np.ndarray  # Mutable B-series coefficients
    symmetry_genes: List[int]  # Immutable symmetry factors
    domain: DomainType
    generation: int = 0
    mutations: int = 0
    
    def mutate(self, mutation_rate: float = 0.1) -> 'KernelGenome':
        """Mutate coefficient genes (symmetry genes are immutable)"""
        mutated_coeffs = self.coefficient_genes.copy()
        
        # Apply mutations to coefficient genes
        mutation_mask = np.random.random(len(mutated_coeffs)) < mutation_rate
        mutations = np.random.normal(0, 0.15, len(mutated_coeffs))
        mutated_coeffs[mutation_mask] += mutations[mutation_mask]
        
        # Ensure coefficients stay in valid range
        mutated_coeffs = np.clip(mutated_coeffs, -2.0, 2.0)
        
        return KernelGenome(
            coefficient_genes=mutated_coeffs,
            symmetry_genes=self.symmetry_genes.copy(),
            domain=self.domain,
            generation=self.generation + 1,
            mutations=self.mutations + int(np.sum(mutation_mask))
        )
    
    def crossover(self, other: 'KernelGenome') -> 'KernelGenome':
        """Genetic crossover with another genome"""
        # Single-point crossover
        crossover_point = np.random.randint(1, len(self.coefficient_genes))
        
        new_coeffs = np.concatenate([
            self.coefficient_genes[:crossover_point],
            other.coefficient_genes[crossover_point:]
        ])
        
        # Inherit symmetry genes from parent with higher fitness
        # (will be evaluated in evolution process)
        new_symmetries = self.symmetry_genes.copy()
        
        return KernelGenome(
            coefficient_genes=new_coeffs,
            symmetry_genes=new_symmetries,
            domain=self.domain,
            generation=max(self.generation, other.generation) + 1,
            mutations=0
        )


@dataclass
class LivingKernel:
    """
    A computational kernel with life-like properties
    """
    genome: KernelGenome
    kernel: ComputationalKernel
    development_stage: DevelopmentStage = DevelopmentStage.EMBRYONIC
    age: int = 0
    fitness: float = 0.0
    
    # Life metrics
    self_generation_count: int = 0
    self_optimization_count: int = 0
    offspring_count: int = 0
    
    def develop(self):
        """Progress through developmental stages"""
        self.age += 1
        
        # Stage transitions based on age
        if self.age > 100:
            self.development_stage = DevelopmentStage.SENESCENT
        elif self.age > 50:
            self.development_stage = DevelopmentStage.MATURE
        elif self.age > 10:
            self.development_stage = DevelopmentStage.JUVENILE
        else:
            self.development_stage = DevelopmentStage.EMBRYONIC
    
    def self_generate(self) -> 'LivingKernel':
        """
        Self-generation through recursive self-composition (chain rule)
        Creates a new kernel by composing with itself
        """
        # Self-composition: apply kernel to its own structure
        new_coeffs = self.genome.coefficient_genes * 1.1  # Amplification
        new_coeffs = np.clip(new_coeffs, -2.0, 2.0)
        
        new_genome = KernelGenome(
            coefficient_genes=new_coeffs,
            symmetry_genes=self.genome.symmetry_genes,
            domain=self.genome.domain,
            generation=self.genome.generation + 1
        )
        
        # Create new kernel
        generator = UniversalKernelGenerator()
        if self.genome.domain == DomainType.PHYSICS:
            new_kernel = generator.generate_physics_kernel(self.kernel.order)
        elif self.genome.domain == DomainType.CONSCIOUSNESS:
            new_kernel = generator.generate_consciousness_kernel(self.kernel.order)
        elif self.genome.domain == DomainType.COMPUTING:
            new_kernel = generator.generate_computing_kernel(self.kernel.order)
        else:
            new_kernel = generator.generate_biology_kernel(self.kernel.order)
        
        new_kernel.b_series_coefficients = new_coeffs
        
        offspring = LivingKernel(
            genome=new_genome,
            kernel=new_kernel,
            development_stage=DevelopmentStage.EMBRYONIC
        )
        
        self.self_generation_count += 1
        return offspring
    
    def self_optimize(self) -> float:
        """
        Self-optimization through iterative grip improvement
        Returns improvement in fitness
        """
        old_fitness = self.fitness
        
        # Optimize grip metric
        generator = UniversalKernelGenerator()
        
        # Domain topology for optimization
        domain_topologies = {
            DomainType.PHYSICS: {
                "symmetries": [2, 4, 6],
                "required_orders": [1, 2, 4],
                "invariants": ["energy"],
                "flow_type": "hamiltonian"
            },
            DomainType.CONSCIOUSNESS: {
                "symmetries": [1, 3, 5],
                "required_orders": [1, 2, 3],
                "invariants": ["attention"],
                "flow_type": "echo_state"
            },
            DomainType.COMPUTING: {
                "symmetries": [1, 2, 4, 8],
                "required_orders": [1, 2, 3, 4],
                "invariants": ["stack_depth"],
                "flow_type": "recursive"
            },
            DomainType.BIOLOGY: {
                "symmetries": [3, 5, 7],
                "required_orders": [1, 2, 3],
                "invariants": ["mass"],
                "flow_type": "metabolic"
            }
        }
        
        topology = domain_topologies.get(self.genome.domain, {})
        optimized_kernel = generator.optimize_grip(self.kernel, topology, iterations=50)
        
        self.kernel = optimized_kernel
        self.genome.coefficient_genes = optimized_kernel.b_series_coefficients
        
        # Update fitness
        self.fitness = self.evaluate_fitness()
        improvement = self.fitness - old_fitness
        
        self.self_optimization_count += 1
        return improvement
    
    def reproduce(self, partner: 'LivingKernel') -> 'LivingKernel':
        """
        Sexual reproduction: crossover and mutation
        """
        # Crossover genomes
        offspring_genome = self.genome.crossover(partner.genome)
        
        # Mutation
        offspring_genome = offspring_genome.mutate(mutation_rate=0.1)
        
        # Create offspring kernel
        generator = UniversalKernelGenerator()
        if offspring_genome.domain == DomainType.PHYSICS:
            offspring_kernel = generator.generate_physics_kernel(self.kernel.order)
        elif offspring_genome.domain == DomainType.CONSCIOUSNESS:
            offspring_kernel = generator.generate_consciousness_kernel(self.kernel.order)
        elif offspring_genome.domain == DomainType.COMPUTING:
            offspring_kernel = generator.generate_computing_kernel(self.kernel.order)
        else:
            offspring_kernel = generator.generate_biology_kernel(self.kernel.order)
        
        offspring_kernel.b_series_coefficients = offspring_genome.coefficient_genes
        
        offspring = LivingKernel(
            genome=offspring_genome,
            kernel=offspring_kernel,
            development_stage=DevelopmentStage.EMBRYONIC
        )
        
        self.offspring_count += 1
        partner.offspring_count += 1
        
        return offspring
    
    def evaluate_fitness(self) -> float:
        """
        Evaluate fitness: grip(0.4) + stability(0.2) + efficiency(0.2) + novelty(0.1) + symmetry(0.1)
        """
        grip = self.kernel.grip_metric
        stability = self.kernel.stability_score
        efficiency = self.kernel.efficiency_score
        
        # Novelty: based on generation and mutations
        novelty = min(1.0, (self.genome.generation * 0.01 + self.genome.mutations * 0.02))
        
        # Symmetry: based on symmetry gene diversity
        symmetry = len(set(self.genome.symmetry_genes)) / max(len(self.genome.symmetry_genes), 1)
        
        fitness = (0.4 * grip + 
                  0.2 * stability + 
                  0.2 * efficiency + 
                  0.1 * novelty + 
                  0.1 * symmetry)
        
        return fitness


class EvolutionEngine:
    """
    Population-based evolutionary optimization of living kernels
    """
    
    def __init__(self, population_size: int = 20, 
                 domain: DomainType = DomainType.COMPUTING,
                 kernel_order: int = 4):
        self.population_size = population_size
        self.domain = domain
        self.kernel_order = kernel_order
        self.population: List[LivingKernel] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def initialize_population(self, seed_kernels: Optional[List[ComputationalKernel]] = None):
        """Initialize population with seed kernels or random generation"""
        self.logger.info(f"Initializing population of {self.population_size} living kernels...")
        
        generator = UniversalKernelGenerator()
        
        if seed_kernels:
            # Use seed kernels
            for seed_kernel in seed_kernels[:self.population_size]:
                genome = KernelGenome(
                    coefficient_genes=seed_kernel.b_series_coefficients.copy(),
                    symmetry_genes=[d.symmetry_factor for d in seed_kernel.elementary_differentials],
                    domain=seed_kernel.domain
                )
                
                living_kernel = LivingKernel(genome=genome, kernel=seed_kernel)
                living_kernel.fitness = living_kernel.evaluate_fitness()
                self.population.append(living_kernel)
        else:
            # Generate random population
            for i in range(self.population_size):
                # Generate kernel based on domain
                if self.domain == DomainType.PHYSICS:
                    kernel = generator.generate_physics_kernel(self.kernel_order)
                elif self.domain == DomainType.CONSCIOUSNESS:
                    kernel = generator.generate_consciousness_kernel(self.kernel_order)
                elif self.domain == DomainType.COMPUTING:
                    kernel = generator.generate_computing_kernel(self.kernel_order)
                else:
                    kernel = generator.generate_biology_kernel(self.kernel_order)
                
                # Add variation
                kernel.b_series_coefficients += np.random.normal(0, 0.1, len(kernel.b_series_coefficients))
                
                genome = KernelGenome(
                    coefficient_genes=kernel.b_series_coefficients,
                    symmetry_genes=[d.symmetry_factor for d in kernel.elementary_differentials],
                    domain=kernel.domain
                )
                
                living_kernel = LivingKernel(genome=genome, kernel=kernel)
                living_kernel.fitness = living_kernel.evaluate_fitness()
                self.population.append(living_kernel)
        
        self.logger.info(f"Population initialized with {len(self.population)} living kernels")
    
    def tournament_selection(self, tournament_size: int = 3) -> LivingKernel:
        """Select parent using tournament selection"""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        winner = max(tournament, key=lambda k: k.fitness)
        return winner
    
    def evolve_generation(self) -> Dict:
        """Evolve population by one generation"""
        self.generation += 1
        self.logger.info(f"Evolving generation {self.generation}...")
        
        # Evaluate all fitness
        for kernel in self.population:
            kernel.fitness = kernel.evaluate_fitness()
            kernel.develop()
        
        # Sort by fitness
        self.population.sort(key=lambda k: k.fitness, reverse=True)
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([k.fitness for k in self.population])
        
        self.best_fitness_history.append(best_fitness)
        
        # Elite preservation (keep top 20%)
        elite_count = max(2, self.population_size // 5)
        next_generation = self.population[:elite_count].copy()
        
        # Generate offspring to fill population
        while len(next_generation) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Reproduction
            offspring = parent1.reproduce(parent2)
            offspring.fitness = offspring.evaluate_fitness()
            
            next_generation.append(offspring)
        
        # Self-optimization for mature kernels
        optimization_count = 0
        for kernel in next_generation:
            if kernel.development_stage == DevelopmentStage.MATURE:
                if np.random.random() < 0.3:  # 30% chance
                    kernel.self_optimize()
                    optimization_count += 1
        
        self.population = next_generation
        
        return {
            "generation": self.generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "elite_count": elite_count,
            "optimization_count": optimization_count,
            "mature_count": sum(1 for k in self.population if k.development_stage == DevelopmentStage.MATURE)
        }
    
    def run_evolution(self, max_generations: int = 50, 
                     fitness_threshold: float = 0.9) -> List[Dict]:
        """Run evolutionary process"""
        self.logger.info(f"Starting evolution for {max_generations} generations...")
        
        history = []
        
        for gen in range(max_generations):
            stats = self.evolve_generation()
            history.append(stats)
            
            self.logger.info(
                f"Gen {stats['generation']}: "
                f"Best={stats['best_fitness']:.4f}, "
                f"Avg={stats['avg_fitness']:.4f}, "
                f"Mature={stats['mature_count']}"
            )
            
            # Check termination
            if stats['best_fitness'] >= fitness_threshold:
                self.logger.info(f"Fitness threshold reached: {stats['best_fitness']:.4f}")
                break
        
        return history
    
    def get_best_kernel(self) -> LivingKernel:
        """Get the best kernel from current population"""
        return max(self.population, key=lambda k: k.fitness)


def demonstrate_ontogenesis():
    """Demonstrate Ontogenesis system"""
    
    logging.basicConfig(level=logging.INFO,
                       format='🌱 [Ontogenesis] %(message)s')
    
    print("🌱 Ontogenesis: Self-Generating Kernel Evolution")
    print("=" * 70)
    print()
    
    # Create evolution engine
    print("🔧 Creating Evolution Engine for Computing Domain...")
    engine = EvolutionEngine(
        population_size=20,
        domain=DomainType.COMPUTING,
        kernel_order=4
    )
    
    # Initialize population
    engine.initialize_population()
    
    print(f"\n📊 Initial Population Stats:")
    print(f"  Population Size: {len(engine.population)}")
    print(f"  Domain: {engine.domain.value}")
    print(f"  Initial Best Fitness: {engine.get_best_kernel().fitness:.4f}")
    
    # Run evolution
    print(f"\n🧬 Running Evolution...")
    history = engine.run_evolution(max_generations=30, fitness_threshold=0.95)
    
    # Final results
    best_kernel = engine.get_best_kernel()
    
    print(f"\n🏆 Evolution Complete!")
    print("=" * 70)
    print(f"  Generations: {engine.generation}")
    print(f"  Best Fitness: {best_kernel.fitness:.4f}")
    print(f"  Kernel Domain: {best_kernel.genome.domain.value}")
    print(f"  Development Stage: {best_kernel.development_stage.value}")
    print(f"  Age: {best_kernel.age}")
    print(f"  Generation: {best_kernel.genome.generation}")
    print(f"  Mutations: {best_kernel.genome.mutations}")
    print(f"  Self-Optimizations: {best_kernel.self_optimization_count}")
    print(f"  Offspring: {best_kernel.offspring_count}")
    
    print(f"\n📈 Fitness History (last 10 generations):")
    for i, fitness in enumerate(engine.best_fitness_history[-10:]):
        print(f"    Gen {len(engine.best_fitness_history)-10+i+1}: {fitness:.4f}")
    
    print("\n" + "=" * 70)
    print("✨ Ontogenesis demonstration complete!")


if __name__ == "__main__":
    demonstrate_ontogenesis()
