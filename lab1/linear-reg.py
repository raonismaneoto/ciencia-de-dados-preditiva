from numpy import *
import matplotlib.pyplot as plt

def plot(rss):
	x = [elem[0] for elem in rss]
	y = [elem[1] for elem in rss]
	plt.scatter(x, y)
	plt.show()

def compute_error_for_given_points(b, m, points):
	total_error = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y-(m*x + b)) **2
	return total_error / float(len(points))

def step_gradient(current_b, current_m, points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N)*(y - ((current_m*x) + current_b))
		m_gradient += -(2/N)*x*(y - ((current_m*x) + current_b))
	new_b = current_b - (learning_rate*b_gradient)
	new_m = current_m - (learning_rate*m_gradient)
	return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, epsilon):
	b = starting_b
	m = starting_m
	rss_for_iterations = []
	iterations = 1
	current_rss = compute_error_for_given_points(b, m, points)
	while current_rss >= epsilon:
		b, m = step_gradient(b, m, array(points), learning_rate)
		rss_for_iterations.append((iterations, current_rss))
		current_rss = compute_error_for_given_points(b, m, points)
		iterations += 1
	plot(rss_for_iterations)
	return [b, m]

def run():
	points = genfromtxt('data.csv', delimiter=",")
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	epsilon = 30
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, epsilon)
	print(b)
	print(m)

run()