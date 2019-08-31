from numpy import *
import matplotlib.pyplot as plt

def plot(rss):
	x = [elem[0] for elem in rss]
	y = [elem[1] for elem in rss]
	plt.scatter(x, y)
	plt.show()

def get_fast(points):
	m = 0
	b = 0
	x_mean = 0
	y_mean = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		x_mean += x
		y_mean += y
	x_mean = x_mean/float(len(points))
	y_mean = y_mean/float(len(points))
	up_side = 0
	down_side = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		up_side += (x - x_mean)*(y-y_mean)
		down_side += (x - x_mean)**2
	m = float(up_side)/float(down_side)
	b = y_mean - m*x_mean
	return [b, m]

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

model = {}

def predict(x, hard=False):
	if hard:
		print(model['bh'] + model['mh']*x)
	else:
		print(model['b'] + model['m']*x)

def run():
	points = genfromtxt('data.csv', delimiter=",")
	learning_rate = 0.001
	initial_b = 0
	initial_m = 0
	# this seems to be a good value in comparison to the other limit method.
	epsilon = 29.9
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, epsilon)
	print(b)
	print(m) 
	[bh, mh] = get_fast(points)
	print(bh)
	print(mh)
	model['b'] = b
	model['m'] = m
	model['bh'] = bh
	model['mh'] = mh

	while True:
		hard = True
		x = int(input("which value do u want to predicct?"))
		predict(x, not hard)

run()