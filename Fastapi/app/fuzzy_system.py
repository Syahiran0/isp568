import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools

# --- Variables ---
input_range = np.arange(0, 101, 1)
attendance = ctrl.Antecedent(input_range, 'attendance')
test_score = ctrl.Antecedent(input_range, 'test_score')
assignment_score = ctrl.Antecedent(input_range, 'assignment_score')
ethics = ctrl.Antecedent(input_range, 'ethics')
cognitive = ctrl.Antecedent(input_range, 'cognitive')
performance = ctrl.Consequent(input_range, 'performance')

for var in [attendance, test_score, assignment_score, ethics, cognitive]:
    var['low'] = fuzz.trimf(input_range, [0, 0, 50])
    var['medium'] = fuzz.trimf(input_range, [25, 50, 75])
    var['high'] = fuzz.trimf(input_range, [50, 100, 100])

performance['weak'] = fuzz.trimf(input_range, [0, 0, 40])
performance['average'] = fuzz.trimf(input_range, [35, 50, 65])
performance['good'] = fuzz.trimf(input_range, [60, 75, 85])
performance['excellent'] = fuzz.trimf(input_range, [80, 100, 100])

"""
# --- 3. Rules ---
rules = [
    # EXCELLENT
    ctrl.Rule(test_score['high'] & assignment_score['high'] & cognitive['high'], performance['excellent'], label='Academic Master'),
    ctrl.Rule(ethics['high'] & attendance['high'] & cognitive['high'], performance['excellent'], label='Professional Leader'),
    ctrl.Rule(test_score['high'] & ethics['high'] & cognitive['high'], performance['excellent'], label='Intellectual Star'),
    ctrl.Rule(attendance['high'] & assignment_score['high'] & test_score['high'], performance['excellent'], label='Hardworking Scholar'),
    ctrl.Rule(test_score['high'] & cognitive['high'] & ethics['high'] & attendance['high'], performance['excellent'], label='Perfect Profile'),
    ctrl.Rule(cognitive['high'] & assignment_score['high'] & ethics['high'], performance['excellent'], label='High Potential'),

    # GOOD
    ctrl.Rule(test_score['high'] & attendance['medium'], performance['good'], label='High Performer/Mid Presence'),
    ctrl.Rule(cognitive['high'] & ethics['medium'], performance['good'], label='Sharp Mind/Standard Prof'),
    ctrl.Rule(attendance['high'] & ethics['high'] & test_score['medium'], performance['good'], label='Dedicated/Mid Results'),
    ctrl.Rule(assignment_score['high'] & cognitive['medium'], performance['good'], label='Reliable Project Worker'),
    ctrl.Rule(test_score['medium'] & assignment_score['medium'] & cognitive['high'], performance['good'], label='Average/High Potential'),
    ctrl.Rule(ethics['high'] & test_score['medium'], performance['good'], label='Ethical/Competent'),
    ctrl.Rule(attendance['high'] & cognitive['medium'], performance['good'], label='Consistent/Capable'),

    # AVERAGE
    ctrl.Rule(attendance['medium'] & test_score['medium'] & ethics['medium'], performance['average'], label='Solid Average'),
    ctrl.Rule(cognitive['medium'] & assignment_score['medium'], performance['average'], label='Standard Performance'),
    ctrl.Rule(test_score['low'] & cognitive['high'] & ethics['high'], performance['average'], label='Talented/Poor Tester'),
    ctrl.Rule(attendance['low'] & test_score['high'] & cognitive['medium'], performance['average'], label='Smart/Low Engagement'),
    ctrl.Rule(ethics['low'] & test_score['high'] & cognitive['high'], performance['average'], label='Brilliant/Unprofessional'),
    ctrl.Rule(assignment_score['low'] & attendance['high'], performance['average'], label='Diligent/Task Struggle'),
    ctrl.Rule(test_score['low'] & assignment_score['high'], performance['average'], label='Practical/No Exams'),
    ctrl.Rule(attendance['medium'] & cognitive['low'] & ethics['high'], performance['average'], label='Effort/Learning Challenge'),

    # WEAK
    ctrl.Rule(test_score['low'] & assignment_score['low'], performance['weak'], label='Academic Failure Risk'),
    ctrl.Rule(attendance['low'] & ethics['low'], performance['weak'], label='Disciplinary Risk'),
    ctrl.Rule(cognitive['low'] & test_score['low'], performance['weak'], label='Low Cognitive/Academic'),
    ctrl.Rule(attendance['low'] & assignment_score['low'], performance['weak'], label='Disengaged'),
    ctrl.Rule(test_score['low'] & ethics['low'], performance['weak'], label='Academic/Ethical Issue'),
    ctrl.Rule(attendance['medium'] & test_score['low'] & assignment_score['low'], performance['weak'], label='Present/Non-Productive'),
    ctrl.Rule(cognitive['low'] & ethics['low'] & attendance['low'], performance['weak'], label='Total Disengagement'),

    # BORDERLINE
    ctrl.Rule(test_score['medium'] & ethics['low'], performance['weak'], label='Unprofessional Mid'),
    ctrl.Rule(attendance['high'] & test_score['low'] & cognitive['low'], performance['weak'], label='Loyal but Struggling'),
    ctrl.Rule(cognitive['high'] & attendance['low'] & assignment_score['low'], performance['average'], label='Underachieving Genius'),
    ctrl.Rule(test_score['high'] & assignment_score['low'] & ethics['low'], performance['average'], label='Capable but Unreliable')
]

PERFORMANCE_CTRL = ctrl.ControlSystem(rules)

"""
antecedents = [attendance, test_score, assignment_score, ethics, cognitive]
terms = ['low', 'medium', 'high']

rules = []

for combo in itertools.product(terms, repeat=5):
    
    antecedent_expr = (attendance[combo[0]] & test_score[combo[1]] & 
                       assignment_score[combo[2]] & ethics[combo[3]] & cognitive[combo[4]])

    high_count = combo.count('high')
    medium_count = combo.count('medium')
    
    if high_count >= 4:
        perf = performance['excellent']
    elif high_count >= 3:
        perf = performance['good']
    elif medium_count >= 3:
        perf = performance['average']
    else:
        perf = performance['weak']
    
    rules.append(ctrl.Rule(antecedent_expr, perf))


# Create control system
PERFORMANCE_CTRL = ctrl.ControlSystem(rules)
# --- Helper Functions ---
def get_performance_level(score: float) -> str:
    if score < 40: return "Weak"
    elif score < 65: return "Average"
    elif score < 85: return "Good"
    return "Excellent"

def compute_rule_strength(rule, inputs):
    """Compute rule strength for given inputs, include very small strengths"""
    strength = 1.0
    for term in rule.antecedent_terms:
        var_name = term.parent.label
        value = inputs[var_name]
        mf = term.mf
        mu = fuzz.interp_membership(np.arange(0, 101, 1), mf, value)
        strength = min(strength, mu)
    return strength

def rule_to_text(rule):
    """Convert a rule to IF...THEN string"""
    conditions = [f"{term.parent.label} IS {term.label}" for term in rule.antecedent_terms]
    antecedent_text = " AND ".join(conditions)
    consequent_text = f"{rule.consequent.label} IS {rule.consequent.terms[0].label}" if hasattr(rule.consequent, 'terms') else str(rule.consequent)
    return f"IF {antecedent_text} THEN {consequent_text}"

# --- Evaluation ---
def evaluate_score(data: dict) -> dict:
    sim = ctrl.ControlSystemSimulation(PERFORMANCE_CTRL)
    sim.input['attendance'] = data['attendance']
    sim.input['test_score'] = data['test_score']
    sim.input['assignment_score'] = data['assignment_score']
    sim.input['ethics'] = data['ethics']
    sim.input['cognitive'] = data['cognitive']
    sim.compute()
    final_score = sim.output['performance']

    # Include all rules with any positive strength
    active_rules = []
    for i, rule in enumerate(rules, 1):
        strength = compute_rule_strength(rule, data)
        if strength > 0.0:  # Include even tiny contributions
            active_rules.append({
                "rule_id": f"Rule {i}",
                "label": getattr(rule, 'label', f"Rule {i}"),
                "logic": rule_to_text(rule),
                "strength": round(float(strength), 4)
            })

    active_rules.sort(key=lambda x: x['strength'], reverse=True)

    return {
        "inputs": data,
        "fuzzy_score": round(final_score, 2),
        "performance_level": get_performance_level(final_score),
        "applied_rules": active_rules
    }

   
