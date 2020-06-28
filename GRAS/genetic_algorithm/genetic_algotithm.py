import random
import math
import os
import logging
import numpy as np
import h5py as hdf5
import time
import datetime

from multi_gpu_build import Build
from meta_plotting import get_4_pvalue

logging.basicConfig(format='[%(funcName)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger()

N_individuals_in_first_init = 500
N_how_much_we_choose = 15

# p-value what we want or bigger
p = 0.05

# connectomes number
N = 104

max_weights = []
low_weights = []

low_delay = 0.5
max_delay = 6

crit_peaks_number = 50

speed = 6

path = '/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/GRAS/multi_gpu_test'


def write_zero(result_folder):
    f = open(f"{result_folder}/time.txt", 'w')
    f.write("0")
    f.close()
    f = open(f"{result_folder}/ampl.txt", 'w')
    f.write("0")
    f.close()
    f = open(f"{result_folder}/2d.txt", 'w')
    f.write("0")
    f.close()


def convert_to_hdf5(result_folder):
    """
    Converts dat files into hdf5 with compression
    Args:
        result_folder (str): folder where is the dat files placed
    """
    # process only files with these muscle names
    for muscle in ["MN_E", "MN_F"]:
        logger.info(f"converting {muscle} dat files to hdf5")
        is_datfile = lambda f: f.endswith(f"{muscle}.dat")
        datfiles = filter(is_datfile, os.listdir(result_folder))

        name = f"gras_{muscle.replace('MN_', '')}_PLT_{speed}cms_40Hz_2pedal_0.025step.hdf5"

        with hdf5.File(f"{result_folder}/{name}", 'w') as hdf5_file:
            for test_index, filename in enumerate(datfiles):
                with open(f"{result_folder}/{filename}") as datfile:
                    try:
                        data = [-float(v) for v in datfile.readline().split()]
                        # check on NaN values (!important)
                        if any(map(np.isnan, data)):
                            logging.info(f"{filename} has NaN... skip")
                            write_zero(result_folder)
                            continue

                        length = len(data)
                        start, end, l = 0, 0, int(length / 10)
                        for i in range(10):
                            end += l
                            arr = data[start:end]
                            start += l
                            hdf5_file.create_dataset(f"#1_112409_PRE_BIPEDAL_normal_21cms_burst7_Ton_{i}.fig", data=arr,
                                                     compression="gzip")
                    except:
                        continue
        # check that hdf5 file was written properly
        with hdf5.File(f"{result_folder}/{name}") as hdf5_file:
            assert all(map(len, hdf5_file.values()))


class Individual:

    def __init__(self):
        self.pvalue = 0.0
        self.pvalue_amplitude = 0.0
        self.pvalue_times = 0.0
        self.peaks_number = 0.0
        self.gen = []
        self.weights = []
        self.delays = []
        self.id = 0
        self.origin = ""

    def __str__(self):
        return f"""p-value = {self.pvalue}, p-value amplitude = {self.pvalue_amplitude}, 
        p-value times = {self.pvalue_times}, peaks number = {self.peaks_number}, origin from {self.origin}\n
        """

    def __eq__(self, other):
        # return self.peaks_number == self.peaks_number
        return self.pvalue == other.pvalue

    def __gt__(self, other):
        # return self.peaks_number > self.peaks_number
        return self.pvalue > other.pvalue

    def __copy__(self):

        new_individual = Individual()

        for g in self.gen:
            new_individual.gen.append(g)

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        return new_individual

    def __len__(self):
        return len(self.gen)

    @staticmethod
    def format_weight(weight):
        return float("{0:.4f}".format(weight))

    @staticmethod
    def format_delay(delay):
        return float("{0:.1f}".format(delay))

    def is_correct(self):
        # return self.peaks_number > 0
        # ~ pvalue_times != 0 and pvalue_amplitude != 0 and pvalue != 0
        return self.pvalue * self.pvalue_amplitude * self.pvalue_times != 0 and self.peaks_number >= crit_peaks_number

    def set_weight(self, min_weight, max_weight):
        self.weights.append(Individual().format_weight(random.uniform(min_weight, max_weight)))
        max_weights.append(max_weight)
        low_weights.append(min_weight)

    def set_delay(self):
        self.delays.append(Individual().format_delay(random.uniform(low_delay, max_delay)))

    # init individual with random weights and delays
    def init(self):

        # Es ~ OMs
        for i in range(5):
            self.set_weight(0.01, 0.5)

        # CVs - OMs
        for i in range(16):
            self.set_weight(0.06, 2)

        # output to Flexor another OM
        for i in range(4):
            self.set_weight(0.001, 0.01)

        # output to eIP
        for i in range(10):
            self.set_weight(0.1, 5)

        for i in range(40):
            self.set_weight(0.05, 1)

        for i in range(15):
            self.set_weight(0.01, 0.2)

        for i in range(2):
            self.set_weight(0.005, 0.06)

        for i in range(4):
            self.set_weight(0.0005, 0.002)

        # eIP ~ MN
        for i in range(2):
            self.set_weight(2, 15)

        for i in range(6):
            self.weights.append(0)

        # init delays
        for i in range(N):
            self.set_delay()

        self.gen = self.weights + self.delays
        self.origin = "first init"


class Data:
    path_to_files = f"{path}/files"
    path_to_dat_folder = f"{path}/dat"

    log_files = [f"{path_to_files}/history.dat", f"{path_to_files}/bests_pvalue.dat",
                 f"{path_to_files}/log.dat", f"{path_to_files}/log_of_bests.dat"]

    files = []

    for i in range(4):
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_21cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_21cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_13.5cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_13.5cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_E_PLT_6cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/gras_F_PLT_6cms_40Hz_2pedal_0.025step.hdf5")
        files.append(f"{path_to_dat_folder}/{i}/a.txt")
        files.append(f"{path_to_dat_folder}/{i}/t.txt")
        files.append(f"{path_to_dat_folder}/{i}/d2.txt")
        files.append(f"{path_to_dat_folder}/{i}/peaks.txt")
        files.append(f"{path_to_dat_folder}/{i}/{i}_MN_E.dat")
        files.append(f"{path_to_dat_folder}/{i}/{i}_MN_F.dat")

    @staticmethod
    def delete(files_arr):
        for file in files_arr:
            if os.path.isfile(file):
                print(f"Deleted {file}")
                os.remove(f"{file}")

    @staticmethod
    def delete_files():
        Data().delete(Data.files)

    @staticmethod
    def delete_all_files():
        Data().delete(Data.log_files + Data.files)


class Population:

    def __init__(self):
        self.individuals = []

    def add_individual(self, individual):
        self.individuals.append(individual)

    def __len__(self):
        return len(self.individuals)

    # init N_individuals_in_first_init individuals for first population
    def first_init(self):
        for i in range(N_individuals_in_first_init):
            individual = Individual()
            individual.init()
            self.add_individual(individual)
            individual.id = i

        print("Population 1 inited")

    # TODO first init for knowing part of weights and delays, it's needed ?


class Fitness:

    # calculate fitness function for instance of Invididual class
    @staticmethod
    def calculate_fitness(individuals, num_population):

        print("CALCULATE FITNESS FUNCTION")

        # converting 4 results data to hdf5
        for i in range(4):
            convert_to_hdf5(f"{path}/dat/{i}")

        # get p-value for 4 individuals
        get_4_pvalue()

        # set p-value to this individuals
        for i in range(len(individuals)):
            individual = individuals[i]

            ampls = open(f'{path}/dat/{i}/a.txt')
            times = open(f'{path}/dat/{i}/t.txt')
            d2 = open(f'{path}/dat/{i}/d2.txt')
            peaks = open(f'{path}/dat/{i}/peaks.txt')

            individual.pvalue_amplitude = float(ampls.readline())
            individual.pvalue_times = float(times.readline())
            individual.pvalue = float(d2.readline())
            individual.peaks_number = float(peaks.readline())

            Fitness.write_pvalue(individual, num_population)

        Data.delete_files()

    @staticmethod
    def write_pvalue(individual, number):

        file = open(f'{path}/files/history.dat', 'a')

        log_str = f"Population {number}:\n {str(individual)}"

        if individual.is_correct():
            log_str += f"Weighs and delays: {' '.join(map(str, individual.gen))}\n"

        file.write(log_str)
        file.close()

    # choose best value of fitness function for population
    @staticmethod
    def best_fitness(current_population):
        return max(current_population.individuals)


class Breeding:

    @staticmethod
    def crossover(individual_1, individual_2):
        length = len(individual_1)

        crossover_point = random.randint(0, length)

        new_individual_1 = Individual()
        new_individual_1.gen = individual_1.gen[:crossover_point] + individual_2.gen[crossover_point:length]
        new_individual_1.weights = new_individual_1.gen[:int(len(new_individual_1) / 2)]
        new_individual_1.delays = new_individual_1.gen[int(len(new_individual_1) / 2):]

        new_individual_2 = Individual()
        new_individual_2.gen = individual_2.gen[:crossover_point] + individual_1.gen[crossover_point:length]
        new_individual_2.weights = new_individual_1.gen[:int(len(new_individual_1) / 2)]
        new_individual_2.delays = new_individual_1.gen[int(len(new_individual_1) / 2):]

        new_individual_1.origin, new_individual_2.origin = "crossover", "crossover"

        return new_individual_1, new_individual_2

    @staticmethod
    def mutation2(individual):

        new_individual = individual.__copy__()

        n = random.randint(1, 2)

        for index, g in enumerate(individual.gen):
            if index % n == n:
                low, high = Breeding.get_low_high(g)
                if index < N:
                    new_individual.weights.append(Individual().format_weight(random.uniform(low, high)))
                else:
                    new_individual.weights.append(Individual().format_delay(random.uniform(low, high)))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin = "mutation2"

        return new_individual

    @staticmethod
    def get_low_high(mean):

        mean = float(mean)

        sigma = abs(mean) / 5
        probability = 0.001
        n = math.sqrt(2 * math.pi * probability * probability * sigma * sigma)

        if n == 0:
            n = probability

        k = math.log(n)
        res = sigma * math.sqrt(-2 * k) if k < 0 else sigma * math.sqrt(2 * k)
        low = mean - res

        if low < 0:
            low = probability / 10

        high = mean + res

        return low, high

    @staticmethod
    def mutation3(individual):

        new_individual = individual.__copy__()

        for index, g in enumerate(individual.gen):
            n = random.randint(0, 100)
            if n < 50:
                low, high = Breeding.get_low_high(g)
                if index < N:
                    new_individual.weights.append(Individual().format_weight(random.uniform(low, high)))
                else:
                    new_individual.weights.append(Individual().format_delay(random.uniform(low, high)))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin = "mutation3"

        return new_individual

    @staticmethod
    def mutation4(individual):

        new_individual = individual.__copy__()

        for index, g in enumerate(individual.gen):
            n = random.randint(0, 100)
            if n < 50:
                m = random.randint(2, 10)
                low, high = g - g / m, g + g / m
                if index < N:
                    new_individual.gen.append(Individual().format_weight(random.uniform(low, high)))
                else:
                    new_individual.weights.append(Individual().format_delay(random.uniform(low, high)))

        new_individual.weights, new_individual.delays = new_individual.gen[:N], new_individual.gen[N:]

        new_individual.origin = "mutation4"

        return new_individual

    @staticmethod
    def mutation(individual):

        new_individual = individual.__copy__()
        mutation_point = random.randint(0, len(individual))

        for index in range(mutation_point):
            mean = new_individual.gen[index]
            low, high = Breeding.get_low_high(mean)
            if index < N:
                new_individual.gen[index] = Individual().format_weight(random.uniform(low, high))
            else:
                new_individual.gen[index] = Individual().format_delay(random.uniform(low, high))

        new_individual.origin = "mutation"

        return new_individual

    # return best N_how_much_we_choose individuals from population
    @staticmethod
    def select(current_population):

        len_current_population = len(current_population)
        logg_string = f"Length current population = {len_current_population}\n"

        newPopulation = Population()

        counter = 0

        # skip incorrect individuals
        for index in range(len_current_population):

            if current_population.individuals[index].is_correct():
                newPopulation.add_individual(current_population.individuals[index])

            else:
                counter += 1

                s = f"Skip individual because {str(current_population.individuals[index])}\n"

                print(s)
                logg_string += s

        logg_string += f"Skiped {counter} individuals\n"

        file = open(f"{path}/files/log.dat", 'a')
        file.write(logg_string)
        file.close()

        # sort this individuals
        arr = sorted(newPopulation.individuals, reverse=True)

        return arr[:N_how_much_we_choose] if len(arr) > N_how_much_we_choose else arr

    @staticmethod
    def calculate_tests_result(current_population, number):

        arr1 = []
        l = len(current_population)
        b = int(l / 4)
        cp = current_population[0:b * 4]
        k = 0

        while True:
            arr = []
            for i in range(4):
                arr.append(cp[k])
                k += 1
            arr1.append(arr)
            if k >= len(cp):
                break

        arr1.append(current_population[b * 4:l])

        for i in range(len(arr1)):
            Build.run_tests(arr1[i])
            time.sleep(0.2)
            Fitness.calculate_fitness(arr1[i], number)


class Evolution:
    terminate_algorithm = False

    @staticmethod
    def start_gen_algorithm(current_population, number):

        Breeding.calculate_tests_result(current_population.individuals, number)
        individuals = Breeding.select(current_population)

        Debug.log_of_bests(individuals, number)

        newPopulation = Population()

        length_population = len(individuals)

        # each with each
        for i in range(length_population):
            individual_1 = individuals[i]
            j = i
            while j + 1 < length_population:
                j += 1
                individual_2 = individuals[j]
                crossover_individual_1, crossover_individual_2 = Breeding.crossover(individual_1, individual_2)
                newPopulation.add_individual(crossover_individual_1)
                newPopulation.add_individual(crossover_individual_2)

        for individual in newPopulation.individuals:
            for i in range(len(individual.gen)):
                individual.gen[i] = float(individual.gen[i])

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation3(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation2(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation4(individual))

        for individual in individuals:
            newPopulation.add_individual(Breeding.mutation(individual))

        for individual in individuals:
            if individual.is_correct():
                newPopulation.add_individual(individual)

        final_population = Population()

        for individual in newPopulation.individuals:

            for j in range(len(individual)):

                if j < N:

                    if individual.gen[j] < low_weights[j]:
                        individual.gen[j] = low_weights[j]
                    if individual.gen[j] > max_weights[j]:
                        individual.gen[j] = max_weights[j]

                else:

                    if individual.gen[j] < low_delay:
                        individual.gen[j] = low_delay
                    if individual.gen[j] > max_delay:
                        individual.gen[j] = max_delay

            final_population.add_individual(individual)

        return final_population


class Debug:

    @staticmethod
    def save(current_population, number):
        best_individual = Fitness.best_fitness(current_population)

        log_str = f"""Population number = {number}\n
        {str(best_individual)}\n
        Weights and delays:\n
        {' '.join(map(str, best_individual.gen))}\n\n
        """

        f = open(f'{path}/files/history.dat', 'a')
        f.write(log_str)

    @staticmethod
    def log_of_bests(individuals, number):
        best_individual = individuals[0]

        file_with_log_of_bests = open(f'{path}/files/log_of_bests.dat', 'a')
        file_with_log_of_bests.write(f"{''.join(map(str, individuals))}\nWas chosen {best_individual}\n\n")
        file_with_log_of_bests.close()

        log_str = f"""In population number {number} best pvalue = {best_individual.pvalue}
        pvalue_ampl = {best_individual.pvalue_amplitude}, pvalue_times = {best_individual.pvalue_times}, 
        origin {best_individual.origin}\n
        {' '.join(map(str, best_individual.gen))}\n
        """

        file = open(f'{path}/files/bests_pvalue.dat', 'a')
        file.write(log_str)
        file.close()


if __name__ == "__main__":

    Data().delete_all_files()

    f = open(f"{path}/files/history.dat", "w")
    f.write(f"{datetime.datetime.now()}\n")
    f.close()

    Build.compile()

    # first initialization to population
    population = Population()
    population.first_init()

    population_number = 1

    while not Evolution.terminate_algorithm:
        print(f"Population number = {population_number}")

        new_population = Evolution.start_gen_algorithm(population, population_number)

        Debug.save(population, population_number)

        population = new_population

        population_number += 1
