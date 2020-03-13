import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import imageio

true_eigenvalues = [167.76622009277344, 89.84439086914062, 106.86729431152344, 51.09503936767578, 44.81565856933594, 43.12944412231445, 38.06986618041992, 30.898088455200195, 25.267065048217773, 24.326675415039062, 19.926258087158203, 16.243072509765625, 15.194313049316406, 15.515100479125977, 14.034353256225586, 12.776123046875, 10.012635231018066, 9.320322036743164, 9.862530708312988, 8.594795227050781]

# Parse File
f = open("data/Lanczos_1203_183831.txt", "r")
f.readline()
times = []
eigenvalues = []
for i in range(100):
	line = f.readline()
	line_s = line.split()
	times.append(line_s[0])
	eig_l = []
	for j in range(1, len(line_s)):
		eig_l.append(float(line_s[j]))
	eigenvalues.append(eig_l)

# Make image frames
images = []
true_x = true_eigenvalues
true_y = [0] * len(true_eigenvalues)
for i in range(1, 100):
	# Plot data
	print(i)
	this_x = eigenvalues[i]
	this_y = [0] * len(eigenvalues[i])
	plt.scatter(true_x, true_y, s=20, color='red')
	plt.scatter(this_x, this_y, s=2, color='black')
	plt.axis([-1, 170, -0.1, 0.1])
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.title("Iteration " + str(i) + ": " + str(times[i]) + " Seconds In")
	imagename = "data/LanczosImages/test_" + str(i+1) + ".png"
	plt.savefig(imagename)
	plt.clf()
	images.append(imageio.imread(imagename))

# Make gif
imageio.mimsave('data/test_lanczos.gif', images)
imageio.mimsave('data/test_lanczos_slow.gif', images, duration=3)


