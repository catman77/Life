import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from IPython.display import HTML
import copy
from typing import List, Dict, Tuple, Set, Union

class GoldenFractalLife:
    """
    Симулятор модифицированной игры Жизнь на основе ДНК и принципов сакральной геометрии.
    """

    # Константы
    PHI = (1 + 5 ** 0.5) / 2  # Золотое сечение, примерно 1.618

    # Типы клеток
    EMPTY = 0         # Пустое пространство
    ZYGOTE = 1        # Зигота/Стволовая клетка
    PENTA = 2         # Пентагональная клетка (рост, энергия)
    HEXA = 3          # Гексагональная клетка (структура, стабильность)
    DIFF_1 = 4        # Дифференцированная клетка типа 1
    DIFF_2 = 5        # Дифференцированная клетка типа 2
    DIFF_3 = 6        # Дифференцированная клетка типа 3
    DIFF_4 = 7        # Дифференцированная клетка типа 4
    DIFF_5 = 8        # Дифференцированная клетка типа 5
    DEAD = 9          # Апоптоз/мертвая клетка

    # Алфавит ДНК
    DNA_ALPHABET = ['A', 'T', 'G', 'C']

    # Карта водородных связей для каждого нуклеотида
    HYDROGEN_BONDS = {'A': 2, 'T': 2, 'G': 3, 'C': 3}

    # Цвета клеток для визуализации
    CELL_COLORS = {
        EMPTY: 'black',
        ZYGOTE: 'gold',
        PENTA: 'red',
        HEXA: 'blue',
        DIFF_1: 'green',
        DIFF_2: 'purple',
        DIFF_3: 'orange',
        DIFF_4: 'cyan',
        DIFF_5: 'magenta',
        DEAD: 'gray'
    }

    def __init__(self, grid_size=50, initial_energy=100.0, mutation_rate=0.01):
        """
        Инициализация симулятора

        Args:
            grid_size: Размер гексагональной сетки
            initial_energy: Начальный уровень энергии организма
            mutation_rate: Вероятность мутации при делении
        """
        self.grid_size = grid_size
        self.initial_energy = initial_energy
        self.mutation_rate = mutation_rate
        
        # Создаем гексагональную сетку
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # ДНК организма - последовательность нуклеотидов
        self.dna = ""
        
        # Интерпретированные команды ДНК
        self.interpreted_commands = []
        
        # Набор активных клеток (их координаты)
        self.cells = set()
        
        # Типы клеток, энергия и другие атрибуты клеток
        self.cell_types = {}     # {(x, y): тип_клетки}
        self.cell_energy = {}    # {(x, y): энергия}
        self.cell_age = {}       # {(x, y): возраст}
        self.cell_dna = {}       # {(x, y): локальная_ДНК}
        
        # Ресурсы сетки (питательные вещества)
        self.resources = np.ones((grid_size, grid_size)) * 0.5
        
        # ЭМ-поле сетки
        self.em_field = np.zeros((grid_size, grid_size))
        
        # Поле морфогенов
        self.morphogens = np.zeros((grid_size, grid_size))
        
        # Возраст организма
        self.age = 0
        
        # История состояний сетки для анимации
        self.history = []
        
        # Параметры для оптимизации ДНК
        self.dna_params = {
            'penta_ratio': 0.5,      # Соотношение пентагональных клеток к общему числу
            'hexa_ratio': 0.5,       # Соотношение гексагональных клеток к общему числу
            'growth_factor': 0.6,    # Фактор роста организма
            'regeneration_factor': 0.5,  # Фактор регенерации
            'resonance_factor': 0.7,     # Фактор резонанса с ЭМ-полем
        }

    def generate_dna(self, dna_length, params=None):
        """
        Генерирует ДНК организма в зависимости от заданных параметров структуры.
        
        Args:
            dna_length: Длина ДНК в нуклеотидах
            params: Словарь параметров структуры ДНК
        """
        if params:
            self.dna_params = params
        
        # Инициализируем ДНК
        dna_list = []
        
        # Вероятности для каждого нуклеотида, исходя из параметров
        # A и T связаны с пентагональными структурами (рост, энергия)
        # G и C связаны с гексагональными структурами (структура, стабильность)
        p_A = self.dna_params['penta_ratio'] / 2.0
        p_T = self.dna_params['penta_ratio'] / 2.0
        p_G = self.dna_params['hexa_ratio'] / 2.0
        p_C = self.dna_params['hexa_ratio'] / 2.0
        
        # Нормализуем вероятности, чтобы они суммировались в 1
        total = p_A + p_T + p_G + p_C
        p_A /= total
        p_T /= total
        p_G /= total
        p_C /= total
        
        # Создаем распределение нуклеотидов
        nucleotide_probs = [p_A, p_T, p_G, p_C]
        
        # Генерируем последовательность ДНК
        for _ in range(dna_length):
            nucleotide = np.random.choice(self.DNA_ALPHABET, p=nucleotide_probs)
            dna_list.append(nucleotide)
        
        # Создаем блоки ДНК, связанные с конкретными функциями
        
        # Блок активации пентагональных клеток
        penta_block_start = int(dna_length * 0.1)
        penta_block_end = int(dna_length * 0.3)
        for i in range(penta_block_start, penta_block_end):
            if random.random() < self.dna_params['growth_factor']:
                dna_list[i % dna_length] = 'A' if random.random() < 0.5 else 'T'
        
        # Блок активации гексагональных клеток
        hexa_block_start = int(dna_length * 0.4)
        hexa_block_end = int(dna_length * 0.6)
        for i in range(hexa_block_start, hexa_block_end):
            if random.random() < (1 - self.dna_params['growth_factor']):
                dna_list[i % dna_length] = 'G' if random.random() < 0.5 else 'C'
        
        # Блок регенерации
        regen_block_start = int(dna_length * 0.65)
        regen_block_end = int(dna_length * 0.75)
        for i in range(regen_block_start, regen_block_end):
            if random.random() < self.dna_params['regeneration_factor']:
                dna_list[i % dna_length] = 'A' if random.random() < 0.7 else 'G'
        
        # Блок резонанса
        resonance_block_start = int(dna_length * 0.8)
        resonance_block_end = int(dna_length * 0.9)
        for i in range(resonance_block_start, resonance_block_end):
            if random.random() < self.dna_params['resonance_factor']:
                dna_list[i % dna_length] = 'T' if random.random() < 0.7 else 'C'
        
        # Соединяем нуклеотиды в строку ДНК
        self.dna = ''.join(dna_list)
        
        # Интерпретируем ДНК в команды
        self._interpret_dna()
        
        return self.dna

    def _interpret_dna(self):
        """
        Интерпретирует ДНК в список команд для выполнения.
        Каждый триплет (3 нуклеотида) кодирует определенную команду.
        """
        # Очищаем предыдущие команды
        self.interpreted_commands = []
        
        # Проверяем, что ДНК не пуста
        if not self.dna:
            # В случае пустой ДНК, создаем базовую "программу" для выживания
            self.interpreted_commands = [
                {'type': 'divide', 'direction': 'random', 'cell_type': self.PENTA},
                {'type': 'absorb_energy', 'amount': 2.0},
                {'type': 'divide', 'direction': 'random', 'cell_type': self.HEXA},
                {'type': 'strengthen', 'amount': 1.0}
            ]
            return
        
        # Разбиваем ДНК на триплеты
        triplets = [self.dna[i:i+3] for i in range(0, len(self.dna), 3)]
        
        # Интерпретируем каждый триплет
        for triplet in triplets:
            # Если триплет неполный, дополняем его
            if len(triplet) < 3:
                triplet = triplet + 'A' * (3 - len(triplet))
            
            # Вычисляем числовое значение триплета для определения команды
            # Каждый нуклеотид имеет свое числовое значение:
            # A=1, T=2, G=3, C=4
            numeric_value = 0
            for nucleotide in triplet:
                if nucleotide == 'A':
                    numeric_value += 1
                elif nucleotide == 'T':
                    numeric_value += 2
                elif nucleotide == 'G':
                    numeric_value += 3
                elif nucleotide == 'C':
                    numeric_value += 4
            
            # Используем числовое значение для определения типа команды
            command_type = numeric_value % 5
            
            # Создаем команду в зависимости от типа
            command = {}
            
            if command_type == 0:
                # Деление клетки
                command['type'] = 'divide'
                
                # Направление деления
                dir_value = (numeric_value // 5) % 6
                directions = ['north', 'northeast', 'southeast', 'south', 'southwest', 'northwest']
                command['direction'] = directions[dir_value] if dir_value < len(directions) else 'random'
                
                # Тип новой клетки
                cell_type_value = (numeric_value // 30) % 5
                cell_types = [self.PENTA, self.HEXA, self.DIFF_1, self.DIFF_2, self.DIFF_3]
                command['cell_type'] = cell_types[cell_type_value] if cell_type_value < len(cell_types) else self.PENTA
            
            elif command_type == 1:
                # Поглощение энергии
                command['type'] = 'absorb_energy'
                
                # Количество энергии
                energy_value = (numeric_value // 5) % 5 + 1
                command['amount'] = energy_value * 0.5  # От 0.5 до 2.5
            
            elif command_type == 2:
                # Укрепление клетки
                command['type'] = 'strengthen'
                
                # Величина укрепления
                strengthen_value = (numeric_value // 5) % 4 + 1
                command['amount'] = strengthen_value * 0.5  # От 0.5 до 2.0
            
            elif command_type == 3:
                # Дифференциация клетки
                command['type'] = 'differentiate'
                
                # В какой тип дифференцировать
                diff_type_value = (numeric_value // 5) % 5 + 1
                diff_types = [self.DIFF_1, self.DIFF_2, self.DIFF_3, self.DIFF_4, self.DIFF_5]
                command['target_type'] = diff_types[diff_type_value-1] if diff_type_value <= len(diff_types) else self.DIFF_1
            
            elif command_type == 4:
                # Регенерация
                command['type'] = 'regenerate'
                
                # Сила регенерации
                regen_value = (numeric_value // 5) % 5 + 1
                command['amount'] = regen_value * 0.5  # От 0.5 до 2.5
            
            # Добавляем команду в список
            self.interpreted_commands.append(command)
        
        # Проверяем, что есть хотя бы одна команда
        if not self.interpreted_commands:
            # В случае пустого списка, создаем базовую "программу" для выживания
            self.interpreted_commands = [
                {'type': 'divide', 'direction': 'random', 'cell_type': self.PENTA},
                {'type': 'absorb_energy', 'amount': 2.0},
                {'type': 'divide', 'direction': 'random', 'cell_type': self.HEXA},
                {'type': 'strengthen', 'amount': 1.0}
            ]

    def initialize_organism(self):
        """Инициализирует начальное состояние организма"""
        # Очищаем текущее состояние
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.cells = set()
        self.cell_types = {}
        self.cell_energy = {}
        self.cell_age = {}
        self.cell_dna = {}
        self.age = 0
        self.history = []
        
        # Сбрасываем окружающую среду
        self.resources = np.ones((self.grid_size, self.grid_size)) * 0.5
        self.em_field = np.zeros((self.grid_size, self.grid_size))
        self.morphogens = np.zeros((self.grid_size, self.grid_size))
        
        # Создаем зародыш (зиготу) в центре сетки
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        
        # Добавляем зиготу
        self.grid[center_x, center_y] = self.ZYGOTE
        self.cells.add((center_x, center_y))
        self.cell_types[(center_x, center_y)] = self.ZYGOTE
        self.cell_energy[(center_x, center_y)] = self.initial_energy
        self.cell_age[(center_x, center_y)] = 0
        
        # Проверяем, есть ли ДНК, если нет, генерируем
        if not self.dna:
            self.generate_dna(dna_length=120)
        
        # Проверяем, интерпретирована ли ДНК
        if not self.interpreted_commands:
            self._interpret_dna()
        
        # Сохраняем начальное состояние
        self.history.append(copy.deepcopy(self.grid))

    def step(self):
        """Выполняет один шаг симуляции жизненного цикла"""
        # Увеличиваем счетчик возраста
        self.age += 1

        # Обновляем окружающую среду
        self._update_environment()

        # Список клеток для обработки (делаем копию, т.к. набор может меняться)
        cells_to_process = list(self.cells)

        # Клетки, которые будут добавлены в этом шаге
        new_cells = set()

        # Клетки, которые будут удалены в этом шаге
        cells_to_remove = set()

        # Обрабатываем каждую клетку
        for cell_pos in cells_to_process:
            # Увеличиваем возраст клетки
            self.cell_age[cell_pos] = self.cell_age.get(cell_pos, 0) + 1

            # Получаем тип клетки
            cell_type = self.cell_types.get(cell_pos, self.EMPTY)

            # Пропускаем мертвые клетки и пустые места
            if cell_type in [self.EMPTY, self.DEAD]:
                continue

            # Получаем энергию клетки
            cell_energy = self.cell_energy.get(cell_pos, 0)

            # Расходуем базовую энергию для поддержания жизни
            cell_energy -= 0.5

            # Получаем локальную ДНК клетки
            cell_dna = self.cell_dna.get(cell_pos, self.dna)

            # Получаем команду из ДНК для текущего шага
            # Проверка на случай пустого списка команд
            if not self.interpreted_commands:
                self._interpret_dna()  # Повторно интерпретируем ДНК
                
            # Если список всё равно пуст, создаем базовую команду
            if not self.interpreted_commands:
                self.interpreted_commands = [{'type': 'absorb_energy', 'amount': 1.0}]
                
            command_index = self.age % len(self.interpreted_commands)
            command = self.interpreted_commands[command_index]

            # Выполняем команду
            if command['type'] == 'divide' and cell_energy > 10:
                # Деление клетки
                neighbor_pos = self._get_random_empty_neighbor(cell_pos)
                if neighbor_pos:
                    # Создаем новую клетку
                    new_cell_type = command.get('cell_type', cell_type)

                    # Расходуем энергию на деление
                    cell_energy -= 8

                    # Добавляем новую клетку
                    self.grid[neighbor_pos] = new_cell_type
                    new_cells.add(neighbor_pos)
                    self.cell_types[neighbor_pos] = new_cell_type
                    self.cell_energy[neighbor_pos] = 5
                    self.cell_age[neighbor_pos] = 0

                    # Проверяем возможность мутации
                    if random.random() < self.mutation_rate:
                        # Создаем мутированную ДНК
                        mutated_dna = self._mutate_dna(cell_dna)
                        self.cell_dna[neighbor_pos] = mutated_dna
                    else:
                        # Наследуем ДНК родителя
                        self.cell_dna[neighbor_pos] = cell_dna

            elif command['type'] == 'absorb_energy':
                # Поглощение энергии из окружающей среды
                amount = command.get('amount', 1.0)
                
                # Поглощаем ресурсы из клетки
                x, y = cell_pos
                available_resources = self.resources[x, y]
                absorbed = min(available_resources, amount)
                
                # Обновляем ресурсы и энергию
                self.resources[x, y] -= absorbed
                cell_energy += absorbed

            elif command['type'] == 'strengthen':
                # Укрепление клетки
                amount = command.get('amount', 1.0)
                
                # Укрепление зависит от типа клетки
                if cell_type == self.PENTA:
                    # Пентагональные клетки лучше укрепляются в энергетическом плане
                    cell_energy += amount * 0.8
                elif cell_type == self.HEXA:
                    # Гексагональные клетки лучше укрепляются структурно
                    cell_energy += amount * 0.5
                    # и живут дольше
                    self.cell_age[cell_pos] -= 1  # Замедляем старение
                else:
                    # Дифференцированные клетки получают среднее укрепление
                    cell_energy += amount * 0.3

            elif command['type'] == 'differentiate':
                # Дифференциация клетки
                target_type = command.get('target_type', self.DIFF_1)
                
                # Дифференциация требует энергии
                if cell_energy > 3:
                    # Расходуем энергию
                    cell_energy -= 3
                    
                    # Изменяем тип клетки
                    self.cell_types[cell_pos] = target_type
                    self.grid[cell_pos] = target_type

            elif command['type'] == 'regenerate':
                # Регенерация
                amount = command.get('amount', 1.0)
                
                # Восстанавливаем соседние клетки
                neighbors = self._get_neighbors(cell_pos)
                for neighbor in neighbors:
                    if neighbor in self.cells:
                        # Увеличиваем энергию соседа
                        self.cell_energy[neighbor] = self.cell_energy.get(neighbor, 0) + amount * 0.4
                        
                        # Если клетка мертвая, возможно оживляем ее
                        if self.cell_types.get(neighbor) == self.DEAD and random.random() < 0.3:
                            # Определяем новый тип клетки на основе контекста
                            surrounding_types = [self.cell_types.get(n) for n in self._get_neighbors(neighbor) if n in self.cells and self.cell_types.get(n) != self.DEAD]
                            
                            if surrounding_types:
                                # Выбираем новый тип на основе окружения
                                new_type = random.choice(surrounding_types)
                                self.cell_types[neighbor] = new_type
                                self.grid[neighbor] = new_type

            # Проверяем энергию клетки после выполнения команды
            if cell_energy <= 0:
                # Клетка погибает из-за отсутствия энергии
                self.cell_types[cell_pos] = self.DEAD
                self.grid[cell_pos] = self.DEAD
                cells_to_remove.add(cell_pos)
            else:
                # Обновляем энергию клетки
                self.cell_energy[cell_pos] = cell_energy

        # Добавляем новые клетки в набор активных клеток
        self.cells.update(new_cells)

        # Удаляем мертвые клетки из набора активных клеток
        self.cells.difference_update(cells_to_remove)

        # Сохраняем текущее состояние
        self.history.append(copy.deepcopy(self.grid))

        # Проверяем, остались ли живые клетки
        alive_cells_count = sum(1 for cell_pos in self.cells if self.cell_types.get(cell_pos) != self.DEAD)
        
        # Возвращаем True, если организм жив, иначе False
        return alive_cells_count > 0

    def _update_environment(self):
        """Обновляет ресурсы и поля окружающей среды"""
        # Пополняем ресурсы (до максимального значения 1.0)
        self.resources += 0.05
        self.resources = np.clip(self.resources, 0, 1.0)
        
        # Обновляем электромагнитное поле
        # Создаем затухающие волны вокруг клеток
        em_field_update = np.zeros_like(self.em_field)
        
        for cell_pos in self.cells:
            x, y = cell_pos
            cell_type = self.cell_types.get(cell_pos)
            
            # Пентагональные клетки создают более сильное ЭМ-поле
            if cell_type == self.PENTA:
                intensity = 0.3
            # Гексагональные клетки создают более стабильное ЭМ-поле
            elif cell_type == self.HEXA:
                intensity = 0.2
            # Стволовые клетки создают сильнейшее поле
            elif cell_type == self.ZYGOTE:
                intensity = 0.5
            # Дифференцированные клетки создают слабое поле
            else:
                intensity = 0.1
            
            # Создаем затухающую волну
            radius = 5  # Радиус влияния
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist <= radius:
                        nx, ny = x + dx, y + dy
                        # Проверяем границы сетки
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            # Интенсивность затухает с расстоянием
                            em_field_update[nx, ny] += intensity * (1 - dist/radius)
        
        # Обновляем ЭМ-поле с затуханием предыдущего состояния
        self.em_field = self.em_field * 0.9 + em_field_update
        
        # Обновляем поле морфогенов
        # Морфогены - сигнальные молекулы, влияющие на развитие организма
        morphogen_update = np.zeros_like(self.morphogens)
        
        for cell_pos in self.cells:
            x, y = cell_pos
            cell_type = self.cell_types.get(cell_pos)
            
            # Разные типы клеток выделяют разные морфогены
            intensity = 0.0
            
            if cell_type == self.ZYGOTE:
                intensity = 0.4  # Стволовые клетки сильно влияют на развитие
            elif cell_type == self.PENTA:
                intensity = 0.3  # Пентагональные клетки - рост
            elif cell_type == self.HEXA:
                intensity = 0.2  # Гексагональные клетки - структура
            elif cell_type in [self.DIFF_1, self.DIFF_2, self.DIFF_3, self.DIFF_4, self.DIFF_5]:
                intensity = 0.1  # Дифференцированные клетки - слабое влияние
            
            # Распространяем морфогены
            radius = 3  # Меньший радиус влияния
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist <= radius:
                        nx, ny = x + dx, y + dy
                        # Проверяем границы сетки
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            # Интенсивность затухает с расстоянием
                            morphogen_update[nx, ny] += intensity * (1 - dist/radius)
        
        # Обновляем поле морфогенов с затуханием
        self.morphogens = self.morphogens * 0.7 + morphogen_update  # Морфогены быстрее рассеиваются

    def _get_random_empty_neighbor(self, cell_pos):
        """
        Возвращает случайную пустую соседнюю клетку.
        
        Args:
            cell_pos: Позиция клетки (x, y)
            
        Returns:
            Позиция пустой соседней клетки или None, если таковой нет
        """
        x, y = cell_pos
        
        # Возможные направления: вверх, вправо, вниз, влево, вверх-вправо, вверх-влево, вниз-вправо, вниз-влево
        directions = [
            (-1, 0),  # Север
            (-1, 1),  # Северо-восток
            (0, 1),   # Восток
            (1, 1),   # Юго-восток
            (1, 0),   # Юг
            (1, -1),  # Юго-запад
            (0, -1),  # Запад
            (-1, -1)  # Северо-запад
        ]
        
        # Перемешиваем направления для случайности
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Проверяем границы сетки
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                # Проверяем, пуста ли клетка
                if self.grid[nx, ny] == self.EMPTY:
                    return (nx, ny)
        
        # Если нет пустых соседей, возвращаем None
        return None

    def _get_neighbors(self, cell_pos):
        """
        Возвращает список всех соседних позиций.
        
        Args:
            cell_pos: Позиция клетки (x, y)
            
        Returns:
            Список позиций соседних клеток
        """
        x, y = cell_pos
        
        # Возможные направления: вверх, вправо, вниз, влево, вверх-вправо, вверх-влево, вниз-вправо, вниз-влево
        directions = [
            (-1, 0),  # Север
            (-1, 1),  # Северо-восток
            (0, 1),   # Восток
            (1, 1),   # Юго-восток
            (1, 0),   # Юг
            (1, -1),  # Юго-запад
            (0, -1),  # Запад
            (-1, -1)  # Северо-запад
        ]
        
        neighbors = []
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Проверяем границы сетки
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        
        return neighbors

    def _mutate_dna(self, dna):
        """
        Мутирует ДНК, случайно изменяя некоторые нуклеотиды.
        
        Args:
            dna: Исходная ДНК
            
        Returns:
            Мутированная ДНК
        """
        # Если ДНК пуста, возвращаем исходную
        if not dna:
            return dna
        
        # Преобразуем ДНК в список для изменения
        dna_list = list(dna)
        
        # Количество мутаций
        num_mutations = max(1, int(len(dna_list) * self.mutation_rate))
        
        # Выбираем случайные позиции для мутации
        mutation_positions = random.sample(range(len(dna_list)), num_mutations)
        
        # Применяем мутации
        for pos in mutation_positions:
            # Выбираем новый нуклеотид, отличный от текущего
            current = dna_list[pos]
            available = [n for n in self.DNA_ALPHABET if n != current]
            dna_list[pos] = random.choice(available)
        
        # Собираем ДНК обратно в строку
        return ''.join(dna_list)

    def analyze_organism(self):
        """
        Анализирует текущее состояние организма.
        
        Returns:
            Словарь с информацией о состоянии организма
        """
        # Подсчитываем количество клеток каждого типа
        cell_counts = {cell_type: 0 for cell_type in range(10)}  # 10 типов клеток
        
        for cell_pos in self.cells:
            cell_type = self.cell_types.get(cell_pos, self.EMPTY)
            cell_counts[cell_type] += 1
        
        # Вычисляем общее количество клеток
        total_cells = sum(cell_counts.values())
        
        # Подсчитываем общую энергию организма
        total_energy = sum(self.cell_energy.values())
        
        # Проверяем, жив ли организм
        is_alive = total_cells > 0 and cell_counts[self.DEAD] < total_cells
        
        # Вычисляем среднюю гармонию организма
        # Гармония определяется отношением пентагональных к гексагональным клеткам
        # и их соответствием золотому сечению
        penta_count = cell_counts[self.PENTA]
        hexa_count = cell_counts[self.HEXA]
        
        if hexa_count > 0:
            penta_hexa_ratio = penta_count / hexa_count
            harmony_score = 1.0 - abs(penta_hexa_ratio - self.PHI) / self.PHI
        else:
            harmony_score = 0.0
        
        # Вычисляем потенциал бессмертия
        if is_alive:
            immortality_potential = harmony_score * 0.3 + \
                                   (penta_count / max(1, total_cells)) * 0.3 + \
                                   (hexa_count / max(1, total_cells)) * 0.3 + \
                                   (self.age / 100) * 0.1  # Чем старше, тем выше потенциал
        else:
            immortality_potential = 0.0
        
        return {
            'age': self.age,
            'is_alive': is_alive,
            'total_cells': total_cells,
            'cell_counts': cell_counts,
            'total_energy': total_energy,
            'harmony_score': harmony_score,
            'immortality_potential': immortality_potential
        }

    def visualize_current_state(self):
        """Визуализирует текущее состояние организма"""
        if not self.grid.any():
            print("Нет данных для визуализации. Инициализируйте организм.")
            return
        
        # Создаем цветовую карту
        cmap = mcolors.ListedColormap([self.CELL_COLORS[i] for i in range(len(self.CELL_COLORS))])
        norm = mcolors.BoundaryNorm(np.arange(len(self.CELL_COLORS) + 1) - 0.5, cmap.N)
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Отображаем сетку
        grid = self.grid
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Добавляем легенду
        cell_types = {
            self.EMPTY: "Пустое пространство",
            self.ZYGOTE: "Зигота/Стволовая клетка",
            self.PENTA: "Пентагональная клетка",
            self.HEXA: "Гексагональная клетка",
            self.DIFF_1: "Дифференцированная клетка 1",
            self.DIFF_2: "Дифференцированная клетка 2",
            self.DIFF_3: "Дифференцированная клетка 3",
            self.DIFF_4: "Дифференцированная клетка 4",
            self.DIFF_5: "Дифференцированная клетка 5",
            self.DEAD: "Мертвая клетка"
        }
        
        patches = [plt.Rectangle((0, 0), 1, 1, fc=self.CELL_COLORS[i]) for i in range(len(self.CELL_COLORS))]
        labels = [cell_types[i] for i in range(len(self.CELL_COLORS))]
        ax.legend(patches, labels, loc='upper right', bbox_to_anchor=(1.4, 1), fontsize='small')
        
        # Настраиваем оси
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)
        
        plt.tight_layout()
        plt.show()

    def visualize(self, step=None, show_fields=False, figsize=(12, 10)):
        """Визуализирует текущее состояние организма и окружающую среду"""
        if step is not None and 0 <= step < len(self.history):
            grid_to_show = self.history[step]
        else:
            grid_to_show = self.grid
        
        if show_fields:
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            
            # Визуализация организма
            ax1 = axs[0, 0]
            self._draw_grid(ax1, grid_to_show)
            ax1.set_title(f"Организм (возраст: {self.age})")
            
            # Визуализация ресурсов
            ax2 = axs[0, 1]
            im2 = ax2.imshow(self.resources, cmap='Greens')
            ax2.set_title("Ресурсы")
            plt.colorbar(im2, ax=ax2)
            
            # Визуализация ЭМ-поля
            ax3 = axs[1, 0]
            im3 = ax3.imshow(self.em_field, cmap='plasma')
            ax3.set_title("Электромагнитное поле")
            plt.colorbar(im3, ax=ax3)
            
            # Визуализация морфогенов
            ax4 = axs[1, 1]
            im4 = ax4.imshow(self.morphogens, cmap='viridis')
            ax4.set_title("Морфогены")
            plt.colorbar(im4, ax=ax4)
        else:
            # Только организм
            fig, ax = plt.subplots(figsize=(10, 8))
            self._draw_grid(ax, grid_to_show)
            ax.set_title(f"Организм (возраст: {self.age})")
        
        plt.tight_layout()
        plt.show()
    
    def _draw_grid(self, ax, grid):
        """Рисует гексагональную сетку с клетками"""
        # Для упрощения визуализации используем обычную сетку
        cmap = mcolors.ListedColormap([self.CELL_COLORS[i] for i in range(len(self.CELL_COLORS))])
        bounds = list(range(len(self.CELL_COLORS) + 1))
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Отображаем сетку
        im = ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Добавляем легенду
        cell_types = {
            self.EMPTY: "Пустое пространство",
            self.ZYGOTE: "Зигота/Стволовая клетка",
            self.PENTA: "Пентагональная клетка",
            self.HEXA: "Гексагональная клетка",
            self.DIFF_1: "Дифференцированная клетка 1",
            self.DIFF_2: "Дифференцированная клетка 2",
            self.DIFF_3: "Дифференцированная клетка 3",
            self.DIFF_4: "Дифференцированная клетка 4",
            self.DIFF_5: "Дифференцированная клетка 5",
            self.DEAD: "Мертвая клетка"
        }
        
        patches = [plt.Rectangle((0, 0), 1, 1, fc=self.CELL_COLORS[i]) for i in range(len(self.CELL_COLORS))]
        labels = [cell_types[i] for i in range(len(self.CELL_COLORS))]
        ax.legend(patches, labels, loc='upper right', bbox_to_anchor=(1.4, 1), fontsize='small')
        
        # Настраиваем оси
        ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.2)

    def visualize_animation(self, interval=200):
        """
        Создает анимацию развития организма.
        
        Args:
            interval: Интервал между кадрами в миллисекундах
        
        Returns:
            Объект анимации
        """
        if not self.history:
            print("Нет данных для анимации. Запустите симуляцию.")
            return None
        
        # Создаем цветовую карту
        cmap = mcolors.ListedColormap([self.CELL_COLORS[i] for i in range(len(self.CELL_COLORS))])
        norm = mcolors.BoundaryNorm(np.arange(len(self.CELL_COLORS) + 1) - 0.5, cmap.N)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Отображаем первый кадр
        im = ax.imshow(self.history[0], cmap=cmap, norm=norm)
        
        # Добавляем заголовок
        title = ax.set_title(f"Шаг: 0")
        
        # Функция обновления кадра
        def update(frame):
            im.set_array(self.history[frame])
            title.set_text(f"Шаг: {frame}")
            return [im, title]
        
        # Создаем анимацию
        ani = animation.FuncAnimation(fig, update, frames=len(self.history),
                                       interval=interval, blit=True)
        
        # Отображаем
        plt.close()  # Закрываем фигуру (отображение будет через HTML)
        
        return ani

    def run_simulation(self, steps=100, visualize_interval=10, save_animation=False):
        """Запускает симуляцию на указанное количество шагов"""
        if not self.dna:
            self.generate_dna(dna_length=120)

        if not self.cells:
            self.initialize_organism()

        for i in range(steps):
            # Выполняем шаг симуляции
            is_alive = self.step()

            # Если организм погиб, прекращаем симуляцию
            if not is_alive:
                print(f"Организм погиб на шаге {i+1}")
                break

            # Выводим информацию о состоянии организма
            if (i+1) % visualize_interval == 0:
                report = self.analyze_organism()
                print(f"Шаг {i+1}:")
                print(f"  Возраст: {report['age']}")
                print(f"  Клеток: {report['total_cells']}")
                print(f"  Пентагональных клеток: {report['cell_counts'][self.PENTA]}")
                print(f"  Гексагональных клеток: {report['cell_counts'][self.HEXA]}")
                print(f"  Гармония: {report['harmony_score']:.2f}")
                print(f"  Потенциал бессмертия: {report['immortality_potential']:.2f}")

        # Создаем анимацию развития
        if save_animation:
            ani = self.visualize_animation()
            return HTML(ani.to_jshtml())
        else:
            # Отображаем финальное состояние
            self.visualize_current_state()
            return None
