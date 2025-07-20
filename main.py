import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict

class GeneticAlgorithmMealPlanner:
    def __init__(self, food_data, days=7, population_size=100, generations=200,
                 crossover_rate=0.8, mutation_rate=0.1, elitism_count=2):
        """
        Initialize the Genetic Algorithm meal planner.

        Args:
            food_data (pd.DataFrame): DataFrame containing food items and their nutritional values
            days (int): Number of days to plan meals for
            population_size (int): Number of solutions in each generation
            generations (int): Number of generations to evolve
            crossover_rate (float): Probability of crossover between parents
            mutation_rate (float): Probability of mutation
            elitism_count (int): Number of best solutions to carry over to next generation
        """
        self.food_data = food_data
        self.days = days
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

        # Nutritional constraints (can be modified)
        self.nutrient_goals = {
            'Calories': {'min': 1800, 'max': 2200, 'weight': 1.0},
            'Protein': {'min': 45, 'max': 70, 'weight': 1.2},
            'Fat': {'min': 60, 'max': 90, 'weight': 1.0},
            'Carbs': {'min': 250, 'max': 350, 'weight': 1.0},
            'Fiber': {'min': 20, 'max': 35, 'weight': 1.1}
        }

        # Meal structure constraints
        self.meal_constraints = {
            'breakfast': {'min_items': 2, 'max_items': 4},
            'lunch': {'min_items': 3, 'max_items': 5},
            'dinner': {'min_items': 3, 'max_items': 5},
            'snack': {'min_items': 1, 'max_items': 2}
        }

        # Other constraints
        self.max_repetition = 5  # Max times a food can be repeated in the plan
        self.min_fruit_veg = 3   # Min servings of fruits/vegetables per day
        self.max_cost = 150      # Max total cost for the plan (in EGP)

        # Preprocess food data
        self.food_ids = self.food_data['FoodID'].values
        self.food_tags = self._extract_food_tags()

    def _extract_food_tags(self):
        """Extract meal tags from food data and create a mapping."""
        tags = {}
        for _, row in self.food_data.iterrows():
            food_id = row['FoodID']
            tag_str = row['MealTags']
            if pd.isna(tag_str):
                tags[food_id] = []
            else:
                tags[food_id] = [t.strip() for t in tag_str.split(',')]
        return tags

    def initialize_population(self):
        """Initialize a random population of meal plans."""
        population = []
        for _ in range(self.population_size):
            plan = []
            for day in range(self.days):
                daily_meals = {}
                for meal_type in self.meal_constraints:
                    min_items = self.meal_constraints[meal_type]['min_items']
                    max_items = self.meal_constraints[meal_type]['max_items']
                    num_items = random.randint(min_items, max_items)

                    # Select food items that are tagged for this meal type
                    valid_foods = [fid for fid in self.food_ids
                                 if meal_type in self.food_tags.get(fid, []) or 'all' in self.food_tags.get(fid, [])]

                    if not valid_foods:
                        valid_foods = self.food_ids  # Fallback if no tagged foods

                    selected_foods = random.choices(valid_foods, k=num_items)
                    daily_meals[meal_type] = selected_foods
                plan.append(daily_meals)
            population.append(plan)
        return population

    def calculate_fitness(self, plan):
        """
        Calculate the fitness of a meal plan.
        Lower fitness is better (we're minimizing deviation from goals).
        """
        total_deviation = 0
        total_cost = 0
        food_counts = defaultdict(int)
        fruit_veg_counts = defaultdict(int)

        # Track daily nutritional values
        daily_nutrients = {day: {nutrient: 0 for nutrient in self.nutrient_goals}
                          for day in range(self.days)}

        # Calculate nutritional values and check constraints
        for day in range(self.days):
            day_fruit_veg = 0
            for meal_type, food_ids in plan[day].items():
                for food_id in food_ids:
                    food = self.food_data[self.food_data['FoodID'] == food_id].iloc[0]

                    # Update nutrient totals
                    for nutrient in self.nutrient_goals:
                        daily_nutrients[day][nutrient] += food[nutrient]

                    # Update cost
                    total_cost += food['Cost']

                    # Track food repetition
                    food_counts[food_id] += 1

                    # Track fruits/vegetables
                    if food['IsFruitVeg'] == 1:
                        day_fruit_veg += 1

            fruit_veg_counts[day] = day_fruit_veg

            # Calculate deviation from nutritional goals for this day
            for nutrient, goals in self.nutrient_goals.items():
                value = daily_nutrients[day][nutrient]
                min_val, max_val = goals['min'], goals['max']
                weight = goals['weight']

                if value < min_val:
                    total_deviation += (min_val - value) * weight
                elif value > max_val:
                    total_deviation += (value - max_val) * weight

        # Add penalty for constraint violations
        penalty = 0

        # 1. Cost constraint violation
        if total_cost > self.max_cost:
            penalty += (total_cost - self.max_cost) * 10

        # 2. Food repetition constraint
        for food_id, count in food_counts.items():
            if count > self.max_repetition:
                penalty += (count - self.max_repetition) * 5

        # 3. Fruit/vegetable constraint
        for day, count in fruit_veg_counts.items():
            if count < self.min_fruit_veg:
                penalty += (self.min_fruit_veg - count) * 3

        # 4. Meal size constraints (already handled in initialization)

        # 5. Forbidden days (not implemented in this example)

        total_fitness = total_deviation + penalty
        return total_fitness

    def selection(self, population, fitness_scores):
        """Tournament selection to choose parents for reproduction."""
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            # Randomly select tournament participants
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            # Select the one with the best (lowest) fitness score
            winner = min(tournament, key=lambda x: x[1])[0]
            selected.append(winner)

        return selected

    def crossover(self, parent1, parent2):
        """Uniform crossover between two parent meal plans."""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        child1 = []
        child2 = []

        for day in range(self.days):
            child1_day = {}
            child2_day = {}

            for meal_type in self.meal_constraints:
                # Randomly choose which parent to take the meal from
                if random.random() < 0.5:
                    child1_day[meal_type] = parent1[day][meal_type].copy()
                    child2_day[meal_type] = parent2[day][meal_type].copy()
                else:
                    child1_day[meal_type] = parent2[day][meal_type].copy()
                    child2_day[meal_type] = parent1[day][meal_type].copy()

            child1.append(child1_day)
            child2.append(child2_day)

        return child1, child2

    def mutation(self, plan):
        """Mutate a meal plan by randomly changing some food items."""
        for day in range(self.days):
            for meal_type in self.meal_constraints:
                if random.random() < self.mutation_rate:
                    # Decide what mutation to perform
                    mutation_type = random.choice(['replace', 'add', 'remove'])
                    current_items = plan[day][meal_type]

                    if mutation_type == 'replace' and current_items:
                        # Replace one random item
                        idx = random.randint(0, len(current_items)-1)
                        valid_foods = [fid for fid in self.food_ids
                                      if meal_type in self.food_tags.get(fid, []) or 'all' in self.food_tags.get(fid, [])]
                        if valid_foods:
                            current_items[idx] = random.choice(valid_foods)

                    elif mutation_type == 'add' and len(current_items) < self.meal_constraints[meal_type]['max_items']:
                        # Add one item
                        valid_foods = [fid for fid in self.food_ids
                                      if meal_type in self.food_tags.get(fid, []) or 'all' in self.food_tags.get(fid, [])]
                        if valid_foods:
                            current_items.append(random.choice(valid_foods))

                    elif mutation_type == 'remove' and len(current_items) > self.meal_constraints[meal_type]['min_items']:
                        # Remove one item
                        if current_items:
                            current_items.pop(random.randint(0, len(current_items)-1))

        return plan

    def evolve(self, population, fitness_scores):
        """Create a new generation through selection, crossover and mutation."""
        new_population = []

        # Elitism: keep the best solutions
        elite_indices = np.argsort(fitness_scores)[:self.elitism_count]
        for idx in elite_indices:
            new_population.append(population[idx])

        # Selection
        selected = self.selection(population, fitness_scores)

        # Crossover and mutation
        for i in range(0, len(selected)-1, 2):
            parent1, parent2 = selected[i], selected[i+1]

            # Crossover
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)

            new_population.extend([child1, child2])

        # If population size is odd, add one more random individual
        if len(new_population) < self.population_size:
            new_population.append(self.initialize_population()[0])

        return new_population[:self.population_size]

    def run(self):
        """Run the genetic algorithm."""
        # Initialize population
        population = self.initialize_population()
        best_fitness_per_generation = []
        avg_fitness_per_generation = []

        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = [self.calculate_fitness(plan) for plan in population]

            # Track statistics
            best_fitness = min(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_fitness_per_generation.append(best_fitness)
            avg_fitness_per_generation.append(avg_fitness)

            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")

            # Evolve to next generation
            population = self.evolve(population, fitness_scores)

        # Get the best solution
        best_index = np.argmin([self.calculate_fitness(plan) for plan in population])
        best_plan = population[best_index]
        best_fitness = self.calculate_fitness(best_plan)

        return best_plan, best_fitness, best_fitness_per_generation, avg_fitness_per_generation

    def evaluate_plan(self, plan):
        """Evaluate a meal plan and print nutritional information."""
        total_cost = 0
        food_counts = defaultdict(int)
        daily_nutrients = {day: {nutrient: 0 for nutrient in self.nutrient_goals}
                          for day in range(self.days)}
        fruit_veg_counts = defaultdict(int)

        for day in range(self.days):
            for meal_type, food_ids in plan[day].items():
                for food_id in food_ids:
                    food = self.food_data[self.food_data['FoodID'] == food_id].iloc[0]

                    # Update nutrient totals
                    for nutrient in self.nutrient_goals:
                        daily_nutrients[day][nutrient] += food[nutrient]

                    # Update cost
                    total_cost += food['Cost']

                    # Track food repetition
                    food_counts[food_id] += 1

                    # Track fruits/vegetables
                    if food['IsFruitVeg'] == 1:
                        fruit_veg_counts[day] += 1

        # Print summary
        print("\nMeal Plan Evaluation:")
        print(f"Total Cost: {total_cost:.2f} EGP")
        print(f"Max Food Repetition: {max(food_counts.values()) if food_counts else 0}")
        print(f"Daily Fruit/Vegetable Servings: {[fruit_veg_counts[d] for d in range(self.days)]}")

        print("\nDaily Nutritional Totals:")
        for day in range(self.days):
            print(f"\nDay {day+1}:")
            for nutrient, goals in self.nutrient_goals.items():
                value = daily_nutrients[day][nutrient]
                min_val, max_val = goals['min'], goals['max']
                status = "OK" if min_val <= value <= max_val else "LOW" if value < min_val else "HIGH"
                print(f"{nutrient}: {value:.1f} ({status})")

        return daily_nutrients, total_cost, food_counts

def load_food_data(file_path):
    """Load food data from CSV file."""
    # This is the data structure matching your image.png
    data = {
        'FoodID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Name': ['Chicken Breast', 'Brown Rice', 'Broccoli', 'Apple', 'Egg',
                'Salmon', 'Quinoa', 'Carrot', 'Banana', 'Yogurt'],
        'Calories': [165, 216, 55, 95, 78, 208, 222, 41, 105, 59],
        'Protein': [31.0, 5.0, 3.7, 0.5, 6.3, 22.0, 8.0, 0.9, 1.3, 10.0],
        'Fat': [3.6, 1.8, 0.6, 0.3, 5.3, 13.0, 3.6, 0.2, 0.4, 0.4],
        'Carbs': [0.0, 45.0, 11.0, 25.0, 0.6, 0.0, 39.0, 10.0, 27.0, 3.6],
        'Fiber': [0.0, 3.5, 2.6, 4.4, 0.0, 0.0, 5.0, 2.8, 3.1, 0.0],
        'Cost': [2.0, 0.5, 0.7, 0.4, 0.3, 3.0, 1.2, 0.25, 0.35, 0.6],
        'IsFruitVeg': [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        'Allergens': ['', '', '', '', '', '', '', '', '', 'milk'],
        'MealTags': ['lunch,dinner', 'lunch,dinner', 'all', 'breakfast,snack',
                    'breakfast', 'lunch,dinner', 'lunch', 'all', 'breakfast,snack', 'breakfast,snack'],
        'ForbiddenDays': ['', '', '', '', 'Friday', '', '', '', '', '']
    }
    return pd.DataFrame(data)

def plot_results(best_fitness, avg_fitness):
    """Plot the convergence of the genetic algorithm."""
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Genetic Algorithm Convergence')
    plt.legend()
    plt.grid()
    plt.show()

def print_meal_plan(plan, food_data):
    """Print the meal plan in a readable format."""
    print("\nGenerated Meal Plan:")
    for day_idx, day_plan in enumerate(plan):
        print(f"\nDay {day_idx + 1}:")
        for meal_type, food_ids in day_plan.items():
            food_names = [food_data[food_data['FoodID'] == fid]['Name'].values[0] for fid in food_ids]
            print(f"{meal_type.capitalize()}: {', '.join(food_names)}")

def test_different_dataset_sizes():
    """Test the algorithm with different dataset sizes."""
    # Load the base dataset
    base_data = load_food_data("food_data.csv")

    # Create different dataset sizes
    small_data = base_data.iloc[:5].copy()  # First 5 items
    medium_data = base_data.iloc[:8].copy()  # First 8 items
    large_data = base_data.copy()            # All 10 items

    datasets = {
        "Small (5 items)": small_data,
        "Medium (8 items)": medium_data,
        "Large (10 items)": large_data
    }

    results = {}

    for name, data in datasets.items():
        print(f"\nRunning GA with {name} dataset...")
        ga = GeneticAlgorithmMealPlanner(data, days=7, population_size=50, generations=100)
        best_plan, best_fitness, best_fitness_history, avg_fitness_history = ga.run()

        results[name] = {
            'best_fitness': best_fitness,
            'best_plan': best_plan,
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }

        print(f"\nResults for {name} dataset:")
        print(f"Best Fitness: {best_fitness:.2f}")
        ga.evaluate_plan(best_plan)

    # Plot results for comparison
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        plt.plot(result['fitness_history'], label=f"{name} (Final: {result['best_fitness']:.2f})")
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('GA Performance with Different Dataset Sizes')
    plt.legend()
    plt.grid()
    plt.show()

    return results

def main():
    # Load food data
    food_data = load_food_data("food_data.csv")

    # Initialize and run GA
    ga = GeneticAlgorithmMealPlanner(food_data, days=7, population_size=100, generations=200)
    best_plan, best_fitness, best_fitness_history, avg_fitness_history = ga.run()

    # Print and plot results
    print(f"\nBest Fitness: {best_fitness:.2f}")
    print_meal_plan(best_plan, food_data)
    ga.evaluate_plan(best_plan)
    plot_results(best_fitness_history, avg_fitness_history)

    # Test with different dataset sizes
    test_different_dataset_sizes()

if __name__ == "__main__":
    main()


!pip install scikit-fuzzy
from skfuzzy import trimf, defuzz, interp_membership
from collections import defaultdict

class FuzzyProteinExtension:
    def __init__(self):
        """Initialize the fuzzy system for protein recommendation."""
        # Universe variables
        self.protein_range = np.arange(30, 100, 1)  # Protein in grams

        # Fuzzy sets for protein intake
        self.low_protein = trimf(self.protein_range, [30, 30, 50])
        self.medium_protein = trimf(self.protein_range, [45, 60, 75])
        self.high_protein = trimf(self.protein_range, [65, 80, 80])

        # Store the original crisp values
        self.original_min = 45
        self.original_max = 70
        self.original_recommended = 50

    def plot_membership_functions(self):
        """Visualize the protein membership functions."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.protein_range, self.low_protein, 'b', linewidth=1.5, label='Low')
        plt.plot(self.protein_range, self.medium_protein, 'g', linewidth=1.5, label='Medium')
        plt.plot(self.protein_range, self.high_protein, 'r', linewidth=1.5, label='High')
        plt.title('Protein Intake Membership Functions')
        plt.ylabel('Membership')
        plt.xlabel('Protein (grams)')
        plt.legend()
        plt.grid()
        plt.show()

    def fuzzify_protein_needs(self, activity_level, body_weight):
        """
        Determine fuzzy protein needs based on user characteristics.

        Args:
            activity_level: 0-1 (0=sendentary, 1=very active)
            body_weight: in kg
        """
        # Rule 1: If activity is low, protein needs are low-medium
        # Rule 2: If activity is medium, protein needs are medium
        # Rule 3: If activity is high, protein needs are medium-high

        # Calculate activity level membership
        act_low = max(0, 1 - activity_level*2) if activity_level < 0.5 else 0
        act_med = 1 - abs(activity_level - 0.5)*2
        act_high = max(0, activity_level*2 - 1) if activity_level > 0.5 else 0

        # Apply rules
        protein_low = min(act_low, 0.7)  # We give some weight to low even if active
        protein_med = max(act_med, 0.3)  # Always some medium
        protein_high = act_high

        # Aggregate the rules
        aggregated = np.fmax(
            np.fmin(protein_low, self.low_protein),
            np.fmax(
                np.fmin(protein_med, self.medium_protein),
                np.fmin(protein_high, self.high_protein)
            )
        )

        # Defuzzify to get crisp protein recommendation
        protein_recommended = defuzz(self.protein_range, aggregated, 'centroid')

        # Calculate range based on recommendation
        protein_min = max(30, protein_recommended - 15)
        protein_max = min(80, protein_recommended + 15)

        return {
            'recommended': protein_recommended,
            'min': protein_min,
            'max': protein_max,
            'aggregated': aggregated
        }

    def plot_fuzzy_result(self, aggregated, protein_recommended):
        """Visualize the fuzzy result and defuzzification."""
        plt.figure(figsize=(8, 4))

        # Plot membership functions
        plt.plot(self.protein_range, self.low_protein, 'b', linewidth=0.5, linestyle=':', alpha=0.5)
        plt.plot(self.protein_range, self.medium_protein, 'g', linewidth=0.5, linestyle=':', alpha=0.5)
        plt.plot(self.protein_range, self.high_protein, 'r', linewidth=0.5, linestyle=':', alpha=0.5)

        # Plot aggregated result
        plt.plot(self.protein_range, aggregated, 'k', linewidth=1.5, label='Aggregated')

        # Plot defuzzified result
        plt.vlines(protein_recommended, 0, 1, linewidth=1.5, color='m', label=f'Defuzzified ({protein_recommended:.1f}g)')

        plt.title('Aggregated Protein Recommendation and Defuzzification')
        plt.ylabel('Membership')
        plt.xlabel('Protein (grams)')
        plt.legend()
        plt.grid()
        plt.show()

class FuzzyGeneticAlgorithmMealPlanner(GeneticAlgorithmMealPlanner):
    def __init__(self, food_data, activity_level=0.5, body_weight=70, **kwargs):
        """
        Extend the GA meal planner with fuzzy protein constraints.

        Args:
            activity_level: 0-1 (0=sendentary, 1=very active)
            body_weight: in kg
        """
        super().__init__(food_data, **kwargs)

        # Store original crisp values before modification
        self.original_min = self.nutrient_goals['Protein']['min']
        self.original_max = self.nutrient_goals['Protein']['max']
        self.original_recommended = self.nutrient_goals['Protein'].get('recommended',
                                                                    (self.original_min + self.original_max)/2)

        # Initialize fuzzy system
        self.fuzzy_system = FuzzyProteinExtension()

        # Get fuzzy protein recommendations
        protein_rec = self.fuzzy_system.fuzzify_protein_needs(activity_level, body_weight)

        # Update nutrient goals with fuzzy protein values
        self.nutrient_goals['Protein']['min'] = protein_rec['min']
        self.nutrient_goals['Protein']['max'] = protein_rec['max']
        self.nutrient_goals['Protein']['recommended'] = protein_rec['recommended']

        # Store fuzzy results for visualization
        self.fuzzy_results = protein_rec

    def plot_fuzzy_protein_info(self):
        """Plot all fuzzy protein information."""
        if not hasattr(self, 'fuzzy_results'):
            print("Fuzzy results not available. Run the planner first.")
            return

        self.fuzzy_system.plot_membership_functions()
        self.fuzzy_system.plot_fuzzy_result(
            self.fuzzy_results['aggregated'],
            self.fuzzy_results['recommended']
        )

        print("\nProtein Intake Recommendations:")
        print(f"Original Crisp Values: Min={self.original_min}g, Max={self.original_max}g, Recommended={self.original_recommended}g")
        print(f"Fuzzy Adjusted Values: Min={self.fuzzy_results['min']:.1f}g, Max={self.fuzzy_results['max']:.1f}g, Recommended={self.fuzzy_results['recommended']:.1f}g")

def compare_fuzzy_vs_crisp():
    """Compare fuzzy and non-fuzzy versions of the GA."""
    # Load food data
    food_data = load_food_data("food_data.csv")

    # Test parameters
    test_cases = [
        {'name': 'Sedentary', 'activity': 0.2, 'weight': 70},
        {'name': 'Moderately Active', 'activity': 0.5, 'weight': 70},
        {'name': 'Very Active', 'activity': 0.8, 'weight': 70}
    ]

    results = []

    for case in test_cases:
        print(f"\nTesting {case['name']} case...")

        # Run crisp version
        print("Running crisp GA...")
        crisp_ga = GeneticAlgorithmMealPlanner(food_data, days=3, population_size=30, generations=50)
        crisp_plan, crisp_fitness, _, _ = crisp_ga.run()

        # Run fuzzy version
        print("Running fuzzy GA...")
        fuzzy_ga = FuzzyGeneticAlgorithmMealPlanner(
            food_data,
            activity_level=case['activity'],
            body_weight=case['weight'],
            days=3,
            population_size=30,
            generations=50
        )
        fuzzy_plan, fuzzy_fitness, _, _ = fuzzy_ga.run()

        # Evaluate plans
        crisp_nutrients, crisp_cost, crisp_counts = crisp_ga.evaluate_plan(crisp_plan)
        fuzzy_nutrients, fuzzy_cost, fuzzy_counts = fuzzy_ga.evaluate_plan(fuzzy_plan)

        # Get average protein intake
        crisp_protein = sum([day['Protein'] for day in crisp_nutrients.values()]) / len(crisp_nutrients)
        fuzzy_protein = sum([day['Protein'] for day in fuzzy_nutrients.values()]) / len(fuzzy_nutrients)

        results.append({
            'case': case['name'],
            'activity_level': case['activity'],
            'crisp_fitness': crisp_fitness,
            'fuzzy_fitness': fuzzy_fitness,
            'crisp_protein': crisp_protein,
            'fuzzy_protein': fuzzy_protein,
            'protein_recommended': fuzzy_ga.fuzzy_results['recommended']
        })

    # Print comparison table
    print("\nComparison Results:")
    print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Case", "Crisp Fitness", "Fuzzy Fitness", "Crisp Protein", "Fuzzy Protein", "Recommended Protein"))
    for res in results:
        print("{:<20} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            res['case'], res['crisp_fitness'], res['fuzzy_fitness'],
            res['crisp_protein'], res['fuzzy_protein'], res['protein_recommended']))

    # Plot comparison
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results))
    width = 0.35

    plt.bar(x - width/2, [r['crisp_fitness'] for r in results], width, label='Crisp')
    plt.bar(x + width/2, [r['fuzzy_fitness'] for r in results], width, label='Fuzzy')

    plt.xlabel('Activity Level')
    plt.ylabel('Fitness Score')
    plt.title('Comparison of Crisp vs Fuzzy GA Performance')
    plt.xticks(x, [r['case'] for r in results])
    plt.legend()
    plt.grid()
    plt.show()

    return results

# Run the comparison
compare_fuzzy_vs_crisp()
