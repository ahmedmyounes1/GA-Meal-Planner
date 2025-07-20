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
