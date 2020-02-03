from PIL import Image
import numpy as np
import random


def evaluate_image():
	given_image = Image.open("input_image.png")
	dimensions = given_image.size  # size is inbuilt attribute
	given_pixels = list(given_image.getdata())  # get list of pixels where each pixel is an rgb value
	pixel_amount = dimensions[0] * dimensions[1]
	return given_pixels, pixel_amount


def generate_individual(pixel_amount):
	random_numbers = np.random.randint(low=0, high=2, size=pixel_amount)  # generate a list of n numbers which are randomly 0 or 1, where n is total number of pixels
	individual = [(0, 0, 0) if number == 0 else (255, 255, 255) for number in random_numbers]  # converts list of random numbers to rgb
	return individual


def evaluate_fitness(pixel_amount, individual, given_pixels):
	fitness = sum(given_pixels[i] != individual[i] for i in range(pixel_amount))
	return fitness

# individual - 784 rgb values
# population - 180 individuals
# a parent is an individual from a population

def crossover_old(parent1, parent2, pixel_amount):
	half_pixel_amount = pixel_amount // 2
	child1 = parent1[:half_pixel_amount] + parent2[half_pixel_amount:]
	child2 = parent2[:half_pixel_amount] + parent1[half_pixel_amount:]
	return child1, child2


def crossover(first_parent, second_parent, pixel_amount):
	random_pixels = random.sample(range(pixel_amount), pixel_amount // 2)  # randomly pick half of the available pixels
	first_child = [None for _ in range(pixel_amount)]  # establish two lists of None values of size of amount of pixels
	second_child = [None for _ in range(pixel_amount)]
	for i in range(pixel_amount):  # take the random half from parent1 and the other half from parent2 to make child1, and vice-versa for child2
		if i in random_pixels:
			first_child[i] = first_parent[i]
			second_child[i] = second_parent[i]
		else:
			first_child[i] = second_parent[i]
			second_child[i] = first_parent[i]
	return first_child, second_child


def breed(population, population_size, pixel_amount):
	sum_to_n = (population_size * (population_size + 1))/2
	probability_distribution = [i/sum_to_n for i in range(population_size, 0, -1)]
	indexes = [*range(population_size)]  #  = [0, 1, 2, 3, 4, 5,... 179]
	selected_individuals = np.random.choice(indexes, population_size, p=probability_distribution) # = [0, 6, 2, 4, 8, 123, 179, 67, 5] (length 180)
	# for every 2 corresponding individuals based on the indexes
	# crossover, add children to new list, which is new population
	children = []
	for i in range(0, population_size, 2):
		parent1 = population[selected_individuals[i]]["individual"]
		parent2 = population[selected_individuals[i+1]]["individual"]
		children += crossover(parent1, parent2, pixel_amount)
	return children


def mutate(population, mutation_rate, pixels_to_select, pixel_amount):
	for individual in population:
		mutated_pixels = random.sample(range(pixel_amount), pixels_to_select)  # randomly pick pixel indexes (out of 784)
		for mutated_pixel in mutated_pixels:
			if individual[mutated_pixel] == (255, 255, 255):
				individual[mutated_pixel] = (0, 0, 0)
			else:
				individual[mutated_pixel] = (255, 255, 255)
	return population


def save_image(individual):
	individual = [individual[(i - 28):i] for i in range(28, 784 + 28, 28)]  # convert from list of size x*y to 2d array of dimensions [x,y]
	individual = np.asarray(individual)  # convert list to numpy array
	new_image = Image.fromarray(individual.astype("uint8"), "RGB")
	new_image.save("result.png", "PNG")  # save image


def main():
	given_pixels, pixel_amount = evaluate_image()
	population_size = 400
	mutation_rate = 0.0036  # 0.3% we are going to mutate each individual by this amount (as opposed to this % of individuals)
	pixels_to_select = round(pixel_amount * mutation_rate)
	population = [generate_individual(pixel_amount) for _ in range(population_size)]
	generation = 0
	fittest_threshold = 100
	fittest_percentage = 0
	while fittest_percentage < fittest_threshold:
		population = [{"individual": individual, "fitness": evaluate_fitness(pixel_amount, individual, given_pixels)} for individual in population]
		population = sorted(population, key=lambda k: k["fitness"])
		highest_fitness = population[0]["fitness"]
		fittest_percentage = 100 - (highest_fitness/pixel_amount * 100)
		population = breed(population, population_size, pixel_amount)
		population = mutate(population, mutation_rate, pixels_to_select, pixel_amount)
		print(f"Generation: {generation}: fittest {fittest_percentage}%")
		generation += 1
	save_image(fittest[0]["individual"])
	# fitness = evaluate_fitness(pixel_amount, individual, given_pixels)


if __name__ == "__main__":
	main()
