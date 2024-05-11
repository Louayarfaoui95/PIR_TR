import schedcat.model.tasks as tasks
import schedcat.sched.edf as edf
import schedcat.sched.rm as rm
import schedcat.sched.dm as dm

# Création de tâches spécifiques, chaque tache modèlise une inférence du modèle 
task_set = [
    tasks.SporadicTask(execution=41.459036, period=25),
    tasks.SporadicTask(execution=73.497, period=20),
    tasks.SporadicTask(execution=88.406421, period=15),
    tasks.SporadicTask(execution=33.051728, period=10),
    tasks.SporadicTask(execution=292.792787, period=5),
]

# Assignation des priorités basés sur la période: Plus grande période, moins la priorité 
for i, t in enumerate(sorted(task_set, key=lambda x: x.period)):
    t.id = i + 1

# Fonction pour tester la schedulabilité sous différents planificateurs
def test_schedulability(tasks, cpu_cores):
    print("Testing schedulability:")
    print(" - EDF Schedulability:", edf.is_schedulable(cpu_cores, tasks))
    print(" - RM Schedulability:", rm.is_schedulable(cpu_cores, tasks))
    print(" - DM Schedulability:", dm.is_schedulable(cpu_cores, tasks))
    total_utilization = sum(t.utilization() for t in tasks)
    print(f"Total Memory Utilization: {total_utilization * 100:.2f}%")

# Exécution 
test_schedulability(task_set, 1)

# Modification des taches pour evaluer les performances
def modify_task_set(factor):
    return [tasks.SporadicTask(execution=t.execution * factor, period=t.period) for t in task_set]

# Tester avec une augmentation de 10% du temps d'exécution
increased_tasks = modify_task_set(1.1)
print("\nTesting with increased execution times by 10%:")
test_schedulability(increased_tasks, 1)

# Tester avec une diminution de 10% du temps d'exécution
decreased_tasks = modify_task_set(0.9)
print("\nTesting with decreased execution times by 10%:")
test_schedulability(decreased_tasks, 1)

# Fonction pour modifier dynamiquement l'ensemble de tâches
def modify_task_set(task_set):
    # Ajout aléatoire d'une nouvelle tâche
    if random.choice([True, False]):
        new_task = tasks.SporadicTask(execution=random.randint(30, 80), period=random.randint(80, 200))
        task_set.append(new_task)
        print(f"Added new task: Execution={new_task.execution}, Period={new_task.period}")
    
    # Retrait aléatoire d'une tâche
    if len(task_set) > 1 and random.choice([True, False]):
        removed_task = random.choice(task_set)
        task_set.remove(removed_task)
        print(f"Removed a task: Execution={removed_task.execution}, Period={removed_task.period}")

    return task_set

# Simulation principale
def run_simulation(iterations=10):
    task_set = create_initial_tasks()
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        task_set = modify_task_set(task_set)
        test_schedulability(task_set)

# Exécuter la simulation
run_simulation(20)
