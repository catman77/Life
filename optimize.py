import random
import numpy as np
from golden_fractal_life import GoldenFractalLife


def optimize_dna(generations=20, population_size=10, simulation_steps=20):
    """
    Улучшенный оптимизатор ДНК для создания долгоживущих организмов.
    
    Args:
        generations: Количество поколений для эволюции
        population_size: Размер популяции в каждом поколении
        simulation_steps: Количество шагов симуляции для оценки
        
    Returns:
        Лучшая найденная ДНК-структура
    """
    best_structures = []
    best_overall_score = 0
    stagnation_counter = 0
    
    # Начальное поколение с большим разнообразием
    population = []
    
    # Создаем начальную популяцию с различными стратегиями
    for i in range(population_size):
        # Варьируем начальные параметры для создания разнообразия
        if i % 3 == 0:
            # Пента-доминантная стратегия
            params = {
                'penta_ratio': random.uniform(0.5, 0.8),       # больше пентагональных (рост, энергия)
                'hexa_ratio': random.uniform(0.2, 0.5),        # меньше гексагональных (структура)
                'growth_factor': random.uniform(0.6, 0.9),     # высокий рост
                'regeneration_factor': random.uniform(0.3, 0.7),
                'resonance_factor': random.uniform(0.3, 0.7)
            }
        elif i % 3 == 1:
            # Гекса-доминантная стратегия
            params = {
                'penta_ratio': random.uniform(0.2, 0.5),        # меньше пентагональных
                'hexa_ratio': random.uniform(0.5, 0.8),         # больше гексагональных (стабильность)
                'growth_factor': random.uniform(0.3, 0.6),      # умеренный рост
                'regeneration_factor': random.uniform(0.5, 0.9), # высокая регенерация
                'resonance_factor': random.uniform(0.3, 0.7)
            }
        else:
            # Золотое сечение стратегия (φ ≈ 1.618)
            phi_ratio = (1 + 5**0.5) / 2
            penta = phi_ratio / (1 + phi_ratio)    # ≈ 0.618
            hexa = 1 / (1 + phi_ratio)             # ≈ 0.382
            params = {
                'penta_ratio': random.uniform(penta * 0.9, penta * 1.1),
                'hexa_ratio': random.uniform(hexa * 0.9, hexa * 1.1),
                'growth_factor': random.uniform(0.4, 0.8),
                'regeneration_factor': random.uniform(0.4, 0.8),
                'resonance_factor': random.uniform(0.4, 0.8)
            }
        
        # Оцениваем параметры с увеличенным количеством шагов симуляции
        score, dna, report = evaluate_dna(params, steps=simulation_steps)
        
        population.append({
            'params': params,
            'score': score,
            'dna': dna,
            'report': report
        })
    
    # Сортируем по оценке (лучшие вначале)
    population.sort(key=lambda x: x['score'], reverse=True)
    
    # Сохраняем лучшего из начальной популяции
    best_structures.append(population[0])
    best_overall_score = population[0]['score']
    print(f"Поколение 0: лучшая оценка {population[0]['score']}")
    
    for gen in range(1, generations):
        # Создаем новое поколение
        new_population = []
        
        # Элитизм - сохраняем лучших 20% особей без изменений
        elite_count = max(1, int(population_size * 0.2))
        new_population.extend(population[:elite_count])
        
        # Используем турнирную селекцию и кроссовер для создания новых особей
        while len(new_population) < population_size * 0.7:
            # Турнирная селекция родителей
            parents = []
            for _ in range(2):
                # Выбираем случайных кандидатов и берем лучшего
                tournament_size = max(2, population_size // 3)
                candidates = random.sample(population, tournament_size)
                winner = max(candidates, key=lambda x: x['score'])
                parents.append(winner)
            
            # Кроссовер - создаем ребенка, комбинируя параметры родителей
            child_params = {}
            for key in parents[0]['params'].keys():
                # С 50% вероятностью берем параметр от первого родителя, иначе от второго
                if random.random() < 0.5:
                    child_params[key] = parents[0]['params'][key]
                else:
                    child_params[key] = parents[1]['params'][key]
            
            # Мутация - с вероятностью 30% немного изменяем каждый параметр
            for key in child_params:
                if random.random() < 0.3:
                    # Амплитуда мутации зависит от "стагнации" - если давно нет улучшений, увеличиваем
                    mutation_scale = 0.1 + 0.1 * min(stagnation_counter, 5)
                    mutation = child_params[key] * random.uniform(1 - mutation_scale, 1 + mutation_scale)
                    child_params[key] = max(0.05, min(0.95, mutation))
            
            # Оцениваем нового потомка
            score, dna, report = evaluate_dna(child_params, steps=simulation_steps)
            
            new_population.append({
                'params': child_params,
                'score': score,
                'dna': dna,
                'report': report
            })
        
        # Добавляем совершенно новые случайные особи для разнообразия (30% от размера)
        random_count = population_size - len(new_population)
        for _ in range(random_count):
            # Используем различные стратегии инициализации
            if stagnation_counter > 3:  # Если долго нет улучшений, увеличиваем разнообразие
                # Совершенно случайные параметры
                params = {
                    'penta_ratio': random.uniform(0.1, 0.9),
                    'hexa_ratio': random.uniform(0.1, 0.9),
                    'growth_factor': random.uniform(0.1, 0.9),
                    'regeneration_factor': random.uniform(0.1, 0.9),
                    'resonance_factor': random.uniform(0.1, 0.9)
                }
            else:
                # Параметры в окрестности лучшего решения, но с большими случайными отклонениями
                best_params = population[0]['params']
                params = {}
                for key, value in best_params.items():
                    # Большой радиус отклонения для исследования
                    params[key] = max(0.05, min(0.95, value * random.uniform(0.5, 1.5)))
            
            # Оцениваем новые параметры
            score, dna, report = evaluate_dna(params, steps=simulation_steps)
            
            new_population.append({
                'params': params,
                'score': score,
                'dna': dna,
                'report': report
            })
        
        # Сортируем новое поколение
        new_population.sort(key=lambda x: x['score'], reverse=True)
        
        # Обновляем популяцию
        population = new_population
        
        # Сохраняем лучшего из текущего поколения
        best_structures.append(population[0])
        print(f"Поколение {gen}: лучшая оценка {population[0]['score']}")
        
        # Проверка на стагнацию (нет улучшения в течение нескольких поколений)
        if population[0]['score'] > best_overall_score:
            best_overall_score = population[0]['score']
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Если стагнация слишком долгая, увеличиваем количество шагов симуляции
        if stagnation_counter >= 5:
            new_steps = simulation_steps + 15
            print(f"Стагнация {stagnation_counter} поколений. Увеличиваем шаги симуляции до {new_steps}")
            simulation_steps = new_steps
            stagnation_counter = 0  # Сбрасываем счетчик
    
    # Находим абсолютно лучшую структуру среди всех поколений
    best_structure = max(best_structures, key=lambda x: x['score'])
    
    # Проводим финальную длительную оценку лучшей структуры
    print("\nПроводим финальную оценку лучшей ДНК на 200 шагах симуляции...")
    final_score, final_dna, final_report = evaluate_dna(best_structure['params'], steps=200)
    
    # Если финальная оценка лучше, обновляем результат
    if final_score > best_structure['score']:
        best_structure = {
            'params': best_structure['params'],
            'score': final_score,
            'dna': final_dna,
            'report': final_report
        }
    
    return best_structure

def evaluate_dna(dna_params, steps=50):
    """
    Оценивает качество параметров ДНК для создания долгоживущего организма.
    
    Args:
        dna_params: Словарь параметров ДНК
        steps: Количество шагов симуляции
        
    Returns:
        Оценка, объект ДНК и отчет о симуляции
    """
    # Создаем симулятор
    simulator = GoldenFractalLife(grid_size=50)
    
    # Генерируем ДНК с заданными параметрами
    simulator.generate_dna(dna_length=120, params=dna_params)
    
    # Инициализируем организм
    simulator.initialize_organism()
    
    # Выполняем симуляцию
    max_cells = 0  # Отслеживаем максимальное количество клеток
    harmony_history = []  # История гармонии за все шаги
    
    for i in range(steps):
        is_alive = simulator.step()
        
        # Собираем данные для анализа стабильности
        current_report = simulator.analyze_organism()
        max_cells = max(max_cells, current_report['total_cells'])
        harmony_history.append(current_report['harmony_score'])
        
        if not is_alive:
            break
    
    # Анализируем результаты
    report = simulator.analyze_organism()
    
    # Вычисляем оценку
    score = 0
    
    if report['is_alive']:
        # Базовые бонусы за выживание и достижения
        score += 500  # Бонус за выживание
        score += report['total_cells'] * 10  # Бонус за количество клеток
        score += report['age'] * 5  # Бонус за возраст
        score += report['harmony_score'] * 200  # Бонус за гармонию
        
        # Бонус за стабильность роста (разница между макс. и текущим кол-вом клеток)
        stability_score = 1.0 - (max_cells - report['total_cells']) / max(1, max_cells)
        score += stability_score * 300
        
        # Бонус за стабильную гармонию на протяжении жизни
        if len(harmony_history) > 10:
            harmony_stability = 1.0 - np.std(harmony_history[-10:])
            score += harmony_stability * 200
        
        # Бонус за соотношение пентагональных и гексагональных клеток близкое к золотому сечению
        phi = (1 + 5**0.5) / 2  # ≈ 1.618
        penta_count = report['cell_counts'][simulator.PENTA]
        hexa_count = report['cell_counts'][simulator.HEXA]
        
        if hexa_count > 0:
            penta_hexa_ratio = penta_count / hexa_count
            phi_similarity = 1.0 - min(abs(penta_hexa_ratio - phi) / phi, 1.0)
            score += phi_similarity * 300
        
        # Экспоненциальный бонус за особо длительное выживание
        if report['age'] > 100:
            score += (report['age'] - 100) ** 1.5
    
    return score, simulator.dna, report