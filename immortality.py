import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.animation as animation
from IPython.display import HTML, display
# Импортируем необходимые классы
from golden_fractal_life import GoldenFractalLife  # Ваш исходный класс симулятора

class AdvancedDNAOptimizer:
    """
    Продвинутый оптимизатор ДНК для максимизации продолжительности жизни организма
    и создания потенциально бессмертных существ.
    """
    
    def __init__(self, population_size=20, num_islands=4,
                 mutation_rate_base=0.1, crossover_rate=0.7, 
                 tournament_size=3, elitism_rate=0.1,
                 max_generations=50, stagnation_limit=10,
                 simulation_steps=300, verbose=True):
        """
        Инициализация оптимизатора ДНК.
        
        Args:
            simulator_class: Класс симулятора (GoldenFractalLife)
            population_size: Размер популяции на каждом острове
            num_islands: Количество независимых "островов" для параллельной эволюции
            mutation_rate_base: Базовая вероятность мутации
            crossover_rate: Вероятность скрещивания
            tournament_size: Размер турнира при отборе
            elitism_rate: Процент лучших особей, сохраняемых без изменений
            max_generations: Максимальное количество поколений
            stagnation_limit: Лимит поколений без улучшения для адаптивных мутаций
            simulation_steps: Максимальное количество шагов для симуляции жизни
            verbose: Подробный вывод информации
        """
        self.simulator_class = GoldenFractalLife
        self.population_size = population_size
        self.num_islands = num_islands
        self.mutation_rate_base = mutation_rate_base
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_rate = elitism_rate
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.simulation_steps = simulation_steps
        self.verbose = verbose
        
        # Для отслеживания прогресса
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.best_lifespans_history = []
        self.best_dna_history = []
        self.mutation_rate_history = []
        
        # Текущее лучшее решение
        self.best_solution = None
        
    def initialize_population(self, structure_params_range=None):
        """
        Инициализирует популяцию ДНК с разнообразными параметрами.
        
        Args:
            structure_params_range: Словарь диапазонов для параметров структуры ДНК
            
        Returns:
            Список островов, каждый из которых содержит список особей (словарей с параметрами и DNA)
        """
        if structure_params_range is None:
            # Расширенные диапазоны для большего разнообразия
            structure_params_range = {
                'penta_ratio': (0.2, 0.8),
                'hexa_ratio': (0.2, 0.8),
                'growth_factor': (0.3, 0.9),
                'regeneration_factor': (0.3, 0.9),
                'resonance_factor': (0.3, 0.9),
                # Новые параметры для повышения разнообразия
                'fractal_complexity': (0.3, 0.9),
                'energy_efficiency': (0.3, 0.9),
                'adaptability': (0.3, 0.9)
            }
        
        # Создаем популяции на разных островах
        islands = []
        
        for island_idx in range(self.num_islands):
            population = []
            
            # Специализируем острова для разных стратегий
            island_specialization = island_idx % 4
            
            for _ in range(self.population_size):
                # Создаем параметры с учетом специализации острова
                if island_specialization == 0:
                    # Остров сбалансированных организмов
                    params = self._create_balanced_params(structure_params_range)
                elif island_specialization == 1:
                    # Остров организмов, фокусирующихся на росте
                    params = self._create_growth_focused_params(structure_params_range)
                elif island_specialization == 2:
                    # Остров организмов, фокусирующихся на регенерации
                    params = self._create_regeneration_focused_params(structure_params_range)
                else:
                    # Остров организмов, фокусирующихся на золотом сечении
                    params = self._create_golden_ratio_focused_params(structure_params_range)
                
                # Создаем симулятор для генерации ДНК
                simulator = self.simulator_class()
                dna = simulator.generate_dna(dna_length=120, params=params)
                
                # Добавляем особь в популяцию
                individual = {
                    'params': params,
                    'dna': dna,
                    'fitness': 0,
                    'lifespan': 0,
                    'harmony': 0,
                    'energy': 0,
                    'cell_count': 0
                }
                
                population.append(individual)
            
            islands.append(population)
        
        return islands
    
    def _create_balanced_params(self, params_range):
        """Создает параметры для сбалансированного организма"""
        params = {}
        for key, (min_val, max_val) in params_range.items():
            # Для сбалансированных организмов используем средние значения с небольшим разбросом
            params[key] = random.uniform(
                (min_val + max_val) / 2 - 0.1, 
                (min_val + max_val) / 2 + 0.1
            )
        return params
    
    def _create_growth_focused_params(self, params_range):
        """Создает параметры для организма, фокусирующегося на росте"""
        params = {}
        for key, (min_val, max_val) in params_range.items():
            if key in ['growth_factor', 'penta_ratio']:
                # Высокие значения для параметров роста
                params[key] = random.uniform(max(0.6, min_val), max_val)
            else:
                params[key] = random.uniform(min_val, max_val)
        return params
    
    def _create_regeneration_focused_params(self, params_range):
        """Создает параметры для организма, фокусирующегося на регенерации"""
        params = {}
        for key, (min_val, max_val) in params_range.items():
            if key in ['regeneration_factor', 'energy_efficiency']:
                # Высокие значения для параметров регенерации
                params[key] = random.uniform(max(0.6, min_val), max_val)
            else:
                params[key] = random.uniform(min_val, max_val)
        return params
    
    def _create_golden_ratio_focused_params(self, params_range):
        """Создает параметры для организма, фокусирующегося на золотом сечении"""
        params = {}
        phi = (1 + 5 ** 0.5) / 2  # Золотое сечение, примерно 1.618
        phi_inverse = 1 / phi      # Примерно 0.618
        
        for key, (min_val, max_val) in params_range.items():
            if key == 'penta_ratio':
                # Значения, близкие к золотому сечению для соотношения пента/гекса
                params[key] = random.uniform(
                    max(min_val, phi_inverse * 0.9), 
                    min(max_val, phi_inverse * 1.1)
                )
            elif key == 'hexa_ratio':
                params[key] = random.uniform(
                    max(min_val, phi_inverse * 0.9), 
                    min(max_val, phi_inverse * 1.1)
                )
            else:
                params[key] = random.uniform(min_val, max_val)
        return params
    
    def evaluate_individual(self, individual, detailed=False):
        """
        Оценивает приспособленность особи, проводя симуляцию ее жизни.
        
        Args:
            individual: Особь для оценки
            detailed: Если True, собирает подробную информацию о жизни особи
            
        Returns:
            Особь с обновленными показателями приспособленности
        """
        # Создаем симулятор
        simulator = self.simulator_class()
        
        # Устанавливаем ДНК
        simulator.dna = individual['dna']
        
        # Инициализируем организм
        simulator.initialize_organism()
        
        # Показатели жизни
        lifespan = 0
        max_energy = 0
        max_cells = 0
        max_harmony = 0
        
        # История развития
        history = []
        
        # Проводим симуляцию
        for i in range(self.simulation_steps):
            # Шаг симуляции
            is_alive = simulator.step()
            lifespan = i + 1
            
            # Если организм погиб, прекращаем симуляцию
            if not is_alive:
                break
            
            # Анализируем состояние организма
            report = simulator.analyze_organism()
            
            # Обновляем максимальные показатели
            max_energy = max(max_energy, report['total_energy'])
            max_cells = max(max_cells, report['total_cells'])
            max_harmony = max(max_harmony, report['harmony_score'])
            
            # Собираем детальную историю
            if detailed:
                history.append({
                    'step': i + 1,
                    'energy': report['total_energy'],
                    'cells': report['total_cells'],
                    'harmony': report['harmony_score']
                })
        
        # Рассчитываем итоговую приспособленность
        # Формула приспособленности: продолжительность жизни + бонусы за энергию, клетки и гармонию
        fitness = lifespan * 100  # Базовая составляющая - продолжительность жизни
        
        # Бонусы за показатели
        energy_bonus = max_energy * 0.5
        cells_bonus = max_cells * 10
        harmony_bonus = max_harmony * 1000
        
        # Особый бонус за выживание определенного количества шагов
        if lifespan >= 100:
            fitness += 10000  # Большой бонус за преодоление порога в 100 шагов
        elif lifespan >= 50:
            fitness += 2000   # Меньший бонус за преодоление порога в 50 шагов
        
        # Бонус за "бессмертие" (достижение максимального времени симуляции)
        if lifespan >= self.simulation_steps:
            fitness += 50000  # Огромный бонус за полную продолжительность симуляции
            
        # Штраф за слишком быструю смерть
        if lifespan < 10:
            fitness = fitness * 0.5  # Штраф для очень нежизнеспособных организмов
        
        # Итоговая приспособленность
        fitness += energy_bonus + cells_bonus + harmony_bonus
        
        # Обновляем показатели особи
        individual['fitness'] = fitness
        individual['lifespan'] = lifespan
        individual['harmony'] = max_harmony
        individual['energy'] = max_energy
        individual['cell_count'] = max_cells
        
        if detailed:
            individual['history'] = history
        
        return individual
    
    def evaluate_population(self, islands, parallel=True):
        """
        Оценивает приспособленность всей популяции.
        
        Args:
            islands: Список островов с популяциями
            parallel: Использовать ли параллельные вычисления
            
        Returns:
            Обновленные острова с оцененными особями
        """
        if parallel:
            # Параллельная оценка всех особей
            all_individuals = []
            island_sizes = []
            
            # Собираем всех особей со всех островов
            for island in islands:
                island_sizes.append(len(island))
                all_individuals.extend(island)
            
            # Используем ProcessPoolExecutor для параллельной оценки
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.evaluate_individual, individual) 
                           for individual in all_individuals]
                
                all_evaluated = []
                for future in as_completed(futures):
                    all_evaluated.append(future.result())
            
            # Восстанавливаем структуру островов
            evaluated_islands = []
            start_idx = 0
            for size in island_sizes:
                island_individuals = all_evaluated[start_idx:start_idx + size]
                evaluated_islands.append(island_individuals)
                start_idx += size
            
            return evaluated_islands
        else:
            # Последовательная оценка
            evaluated_islands = []
            
            for island in islands:
                evaluated_island = []
                for individual in island:
                    evaluated_individual = self.evaluate_individual(individual)
                    evaluated_island.append(evaluated_individual)
                evaluated_islands.append(evaluated_island)
            
            return evaluated_islands
    
    def select_parent(self, population):
        """
        Выбирает родителя с помощью турнирного отбора.
        
        Args:
            population: Список особей
            
        Returns:
            Выбранная особь
        """
        # Турнирный отбор: выбираем несколько случайных особей и берем лучшую
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def crossover(self, parent1, parent2):
        """
        Выполняет скрещивание между двумя родителями.
        
        Args:
            parent1, parent2: Родительские особи
            
        Returns:
            Две новые особи-потомка
        """

        def chunkstring(string, length):
            return list(string[0+i:length+i] for i in range(0, len(string), length))

        # Скрещивание параметров
        if random.random() < self.crossover_rate:
            # Определяем точку кроссинговера для параметров
            param_keys = list(parent1['params'].keys())
            crossover_point = random.randint(1, len(param_keys) - 1)
            
            # Создаем новые параметры для потомков
            child1_params = {}
            child2_params = {}
            
            for i, key in enumerate(param_keys):
                if i < crossover_point:
                    child1_params[key] = parent1['params'][key]
                    child2_params[key] = parent2['params'][key]
                else:
                    child1_params[key] = parent2['params'][key]
                    child2_params[key] = parent1['params'][key]
                    
            # Скрещивание ДНК - сначала разбиваем на триплеты
            # parent1_triplets = parent1['dna'].split('-')
            # parent2_triplets = parent2['dna'].split('-')
            parent1_triplets = chunkstring(parent1['dna'], 3)
            parent2_triplets = chunkstring(parent1['dna'], 3)
            
            # Выбираем точку кроссинговера для ДНК
            dna_crossover_point = random.randint(1, min(len(parent1_triplets), len(parent2_triplets)) - 1)
            
            # Создаем новые ДНК для потомков
            child1_triplets = parent1_triplets[:dna_crossover_point] + parent2_triplets[dna_crossover_point:]
            child2_triplets = parent2_triplets[:dna_crossover_point] + parent1_triplets[dna_crossover_point:]
            
            child1_dna = '-'.join(child1_triplets)
            child2_dna = '-'.join(child2_triplets)
        else:
            # Без скрещивания - копируем родителей
            child1_params = copy.deepcopy(parent1['params'])
            child2_params = copy.deepcopy(parent2['params'])
            child1_dna = parent1['dna']
            child2_dna = parent2['dna']
        
        # Создаем потомков
        child1 = {
            'params': child1_params,
            'dna': child1_dna,
            'fitness': 0,
            'lifespan': 0,
            'harmony': 0,
            'energy': 0,
            'cell_count': 0
        }
        
        child2 = {
            'params': child2_params,
            'dna': child2_dna,
            'fitness': 0,
            'lifespan': 0,
            'harmony': 0,
            'energy': 0,
            'cell_count': 0
        }
        
        return child1, child2
    
    def mutate(self, individual, mutation_rate):
        """
        Мутирует особь с заданной вероятностью.
        
        Args:
            individual: Особь для мутации
            mutation_rate: Базовая вероятность мутации
            
        Returns:
            Мутированная особь
        """
        # Копируем особь перед мутацией
        mutated = copy.deepcopy(individual)
        
        # 1. Мутация параметров
        for key in mutated['params']:
            if random.random() < mutation_rate:
                # Применяем гауссову мутацию (нормальное распределение) для плавных изменений
                current_value = mutated['params'][key]
                
                # Стандартное отклонение зависит от текущего значения
                std_dev = current_value * 0.2  # 20% от текущего значения
                
                # Генерируем новое значение с нормальным распределением
                new_value = random.normalvariate(current_value, std_dev)
                
                # Ограничиваем значение параметра в разумных пределах
                new_value = max(0.1, min(0.9, new_value))
                
                mutated['params'][key] = new_value
        
        # 2. Мутация ДНК
        # Разбиваем ДНК на триплеты для удобства мутации
        triplets = mutated['dna'].split('-')
        
        # Разные типы мутаций ДНК
        
        # Мутация 1: Замена нуклеотидов
        for i in range(len(triplets)):
            for j in range(len(triplets[i])):
                if random.random() < mutation_rate:
                    # Замена нуклеотида на другой
                    nucleotides = 'ATGC'
                    current = triplets[i][j]
                    options = nucleotides.replace(current, '')  # Все нуклеотиды, кроме текущего
                    new_nucleotide = random.choice(options)
                    
                    # Создаем новый триплет с замененным нуклеотидом
                    triplets[i] = triplets[i][:j] + new_nucleotide + triplets[i][j+1:]
        
        # Мутация 2: Инверсия участка ДНК (с меньшей вероятностью)
        if random.random() < mutation_rate * 0.3 and len(triplets) > 3:
            # Выбираем случайный участок для инверсии
            start = random.randint(0, len(triplets) - 3)
            end = random.randint(start + 1, min(start + 5, len(triplets) - 1))
            
            # Инвертируем участок
            triplets[start:end+1] = reversed(triplets[start:end+1])
        
        # Мутация 3: Дупликация участка ДНК (с еще меньшей вероятностью)
        if random.random() < mutation_rate * 0.2 and len(triplets) > 2:
            # Выбираем случайный участок для дупликации
            start = random.randint(0, len(triplets) - 2)
            end = random.randint(start, min(start + 3, len(triplets) - 1))
            
            # Дублируем участок (вставляем копию после оригинала)
            segment = triplets[start:end+1]
            insert_pos = end + 1 if end + 1 < len(triplets) else len(triplets)
            triplets = triplets[:insert_pos] + segment + triplets[insert_pos:]
        
        # Мутация 4: Добавление нового триплета (с очень малой вероятностью)
        if random.random() < mutation_rate * 0.1:
            # Создаем новый случайный триплет
            new_triplet = ''.join(random.choice('ATGC') for _ in range(3))
            
            # Добавляем его в случайную позицию
            insert_pos = random.randint(0, len(triplets))
            triplets.insert(insert_pos, new_triplet)
        
        # Мутация 5: Удаление триплета (с очень малой вероятностью)
        if random.random() < mutation_rate * 0.1 and len(triplets) > 4:
            # Удаляем случайный триплет
            del_pos = random.randint(0, len(triplets) - 1)
            triplets.pop(del_pos)
        
        # Собираем ДНК обратно
        mutated['dna'] = '-'.join(triplets)
        
        return mutated
    
    def evolve_island(self, island, generation, stagnation_count):
        """
        Эволюционирует отдельный остров популяции.
        
        Args:
            island: Остров с популяцией
            generation: Текущее поколение
            stagnation_count: Число поколений без улучшения
            
        Returns:
            Новый остров с потомками
        """
        # Сортируем особей по приспособленности
        island.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Определяем количество элитных особей, которые переходят без изменений
        num_elites = max(1, int(len(island) * self.elitism_rate))
        elites = island[:num_elites]
        
        # Создаем новое поколение
        new_island = []
        
        # Добавляем элитных особей
        new_island.extend(elites)
        
        # Адаптивная скорость мутации в зависимости от стагнации
        adaptive_mutation_rate = self.mutation_rate_base * (1 + stagnation_count / self.stagnation_limit)
        adaptive_mutation_rate = min(0.5, adaptive_mutation_rate)  # Ограничиваем макс. скорость мутации
        
        # Создаем потомков до заполнения нового острова
        while len(new_island) < len(island):
            # Выбираем родителей с помощью турнирного отбора
            parent1 = self.select_parent(island)
            parent2 = self.select_parent(island)
            
            # Скрещивание
            child1, child2 = self.crossover(parent1, parent2)
            
            # Мутация
            child1 = self.mutate(child1, adaptive_mutation_rate)
            child2 = self.mutate(child2, adaptive_mutation_rate)
            
            # Добавляем потомков (если есть место)
            if len(new_island) < len(island):
                new_island.append(child1)
            if len(new_island) < len(island):
                new_island.append(child2)
        
        return new_island
    
    def migrate_between_islands(self, islands, migration_rate=0.1):
        """
        Выполняет миграцию особей между островами.
        
        Args:
            islands: Список островов
            migration_rate: Доля особей, мигрирующих между островами
            
        Returns:
            Обновленные острова после миграции
        """
        if len(islands) <= 1:
            return islands
        
        # Количество мигрантов с каждого острова
        num_migrants = max(1, int(len(islands[0]) * migration_rate))
        
        # Копируем острова перед миграцией
        new_islands = [island.copy() for island in islands]
        
        # Для каждого острова выбираем лучших особей для миграции
        migrants = []
        for island in islands:
            # Сортируем по приспособленности и выбираем лучших
            sorted_island = sorted(island, key=lambda x: x['fitness'], reverse=True)
            island_migrants = sorted_island[:num_migrants]
            migrants.append(island_migrants)
        
        # Выполняем миграцию: остров i получает мигрантов с острова i+1
        for i in range(len(new_islands)):
            # Находим индекс острова-донора
            donor_idx = (i + 1) % len(islands)
            
            # Выбираем случайных особей для замены
            replace_indices = random.sample(range(len(new_islands[i])), num_migrants)
            
            # Заменяем особей мигрантами
            for j, migrant_idx in enumerate(replace_indices):
                new_islands[i][migrant_idx] = copy.deepcopy(migrants[donor_idx][j])
        
        return new_islands
    
    def calculate_population_diversity(self, population):
        """
        Рассчитывает разнообразие в популяции.
        
        Args:
            population: Список особей
            
        Returns:
            Показатель разнообразия (0-1)
        """
        if not population:
            return 0
        
        # Рассчитываем разнообразие на основе ДНК
        # Сначала собираем все триплеты ДНК
        all_triplets = set()
        for individual in population:
            triplets = individual['dna'].split('-')
            all_triplets.update(triplets)
        
        # Подсчитываем количество уникальных триплетов для каждой особи
        unique_triplet_counts = []
        for individual in population:
            triplets = set(individual['dna'].split('-'))
            unique_triplet_counts.append(len(triplets))
        
        # Среднее количество уникальных триплетов
        avg_unique_triplets = sum(unique_triplet_counts) / len(population)
        
        # Нормализуем по общему количеству уникальных триплетов
        diversity = avg_unique_triplets / len(all_triplets) if all_triplets else 0
        
        return diversity
    
    def optimize(self, structure_params_range=None):
        """
        Запускает процесс оптимизации ДНК.
        
        Args:
            structure_params_range: Диапазоны параметров для инициализации
            
        Returns:
            Лучшее найденное решение
        """
        # Инициализируем популяцию
        islands = self.initialize_population(structure_params_range)
        
        # Оцениваем начальную популяцию
        islands = self.evaluate_population(islands)
        
        # Найти лучшую особь на всех островах
        best_individual = None
        best_fitness = -float('inf')
        
        for island in islands:
            for individual in island:
                if individual['fitness'] > best_fitness:
                    best_fitness = individual['fitness']
                    best_individual = copy.deepcopy(individual)
        
        # Сохраняем лучшее решение
        self.best_solution = best_individual
        
        # Счетчик поколений без улучшения
        stagnation_count = 0
        
        # Запускаем эволюцию
        for generation in range(self.max_generations):
            # Эволюция каждого острова
            for i in range(len(islands)):
                islands[i] = self.evolve_island(islands[i], generation, stagnation_count)
            
            # Миграция между островами (каждые 5 поколений)
            if generation % 5 == 0 and generation > 0:
                islands = self.migrate_between_islands(islands)
            
            # Оценка новой популяции
            islands = self.evaluate_population(islands)
            
            # Поиск лучшего решения во всех островах
            best_fitness = -float('inf')
            for island in islands:
                for individual in island:
                    if individual['fitness'] > best_fitness:
                        best_fitness = individual['fitness']
                        best_individual = copy.deepcopy(individual)
                        stagnation_count = 0  # Сбрасываем счетчик стагнации
            
            # Проверяем, было ли улучшение
            if best_individual == self.best_solution:
                stagnation_count += 1
            else:
                stagnation_count = 0
            
            # Сохраняем лучшее решение
            self.best_solution = best_individual
            
            # Собираем статистику для визуализации
            overall_population = [ind for island in islands for ind in island]
            avg_fitness = sum(ind['fitness'] for ind in overall_population) / len(overall_population)
            diversity = self.calculate_population_diversity(overall_population)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            self.best_lifespans_history.append(best_individual['lifespan'])
            self.best_dna_history.append(best_individual['dna'])
            self.mutation_rate_history.append(self.mutation_rate_base * (1 + stagnation_count / self.stagnation_limit))
            
            # Выводим информацию о прогрессе
            if self.verbose and (generation % max(1, self.max_generations // 10) == 0 or generation == self.max_generations - 1):
                print(f"Поколение {generation+1}/{self.max_generations}:")
                print(f"  Лучшая приспособленность: {best_fitness:.2f}")
                print(f"  Средняя приспособленность: {avg_fitness:.2f}")
                print(f"  Разнообразие популяции: {diversity:.2f}")
                print(f"  Продолжительность жизни: {best_individual['lifespan']} шагов")
                print(f"  Стагнация: {stagnation_count} поколений")
                print(f"  Скорость мутации: {self.mutation_rate_base * (1 + stagnation_count / self.stagnation_limit):.3f}")
                print()
        
        # Возвращаем лучшее найденное решение
        return self.best_solution
    
    def visualize_optimization_progress(self):
        """Визуализирует прогресс оптимизации"""
        if not self.best_fitness_history:
            print("Нет данных для визуализации. Запустите оптимизацию сначала.")
            return
        
        generations = range(1, len(self.best_fitness_history) + 1)
        
        # Создаем фигуру с несколькими графиками
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # График 1: Приспособленность (лучшая и средняя)
        axs[0].plot(generations, self.best_fitness_history, 'b-', label='Лучшая приспособленность')
        axs[0].plot(generations, self.avg_fitness_history, 'r--', label='Средняя приспособленность')
        axs[0].set_xlabel('Поколение')
        axs[0].set_ylabel('Приспособленность')
        axs[0].set_title('Динамика приспособленности')
        axs[0].legend()
        axs[0].grid(True)
        
        # График 2: Продолжительность жизни и разнообразие
        ax2 = axs[1]
        ax2.plot(generations, self.best_lifespans_history, 'g-', label='Продолжительность жизни')
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Продолжительность жизни (шаги)')
        ax2.set_title('Продолжительность жизни лучшего организма')
        ax2.grid(True)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(generations, self.diversity_history, 'm--', label='Разнообразие')
        ax2_twin.set_ylabel('Разнообразие (0-1)')
        
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # График 3: Скорость мутации
        axs[2].plot(generations, self.mutation_rate_history, 'c-', label='Скорость мутации')
        axs[2].set_xlabel('Поколение')
        axs[2].set_ylabel('Скорость мутации')
        axs[2].set_title('Адаптивная скорость мутации')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def simulate_best_solution(self, detailed=True, save_animation=True):
        """
        Запускает подробную симуляцию лучшего найденного решения.
        
        Args:
            detailed: Собирать подробную информацию о развитии
            save_animation: Сохранить анимацию развития организма
            
        Returns:
            Результаты подробной симуляции
        """
        if self.best_solution is None:
            print("Лучшее решение не найдено. Запустите оптимизацию сначала.")
            return None
        
        print("Проводится подробная симуляция лучшего организма...")
        
        # Создаем симулятор
        simulator = self.simulator_class()
        
        # Устанавливаем ДНК
        simulator.dna = self.best_solution['dna']
        
        # Инициализируем организм
        simulator.initialize_organism()
        
        # Запускаем подробную симуляцию
        simulator.run_simulation(
            steps=self.simulation_steps,
            visualize_interval=max(1, self.simulation_steps // 10),
            save_animation=save_animation
        )
        
        # Получаем итоговый отчет
        final_report = simulator.analyze_organism()
        
        print("\nИтоговые результаты для лучшего организма:")
        print(f"Продолжительность жизни: {self.best_solution['lifespan']} шагов")
        print(f"Общая приспособленность: {self.best_solution['fitness']:.2f}")
        print(f"Максимальное количество клеток: {self.best_solution['cell_count']}")
        print(f"Максимальная гармония: {self.best_solution['harmony']:.2f}")
        print(f"Максимальная энергия: {self.best_solution['energy']:.2f}")
        print("\nПараметры ДНК:")
        for key, value in self.best_solution['params'].items():
            print(f"- {key}: {value:.3f}")
        
        return {
            'simulator': simulator,
            'solution': self.best_solution,
            'final_report': final_report
        }
    