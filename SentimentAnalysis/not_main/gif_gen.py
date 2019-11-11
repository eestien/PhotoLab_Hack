import imageio
images = []
filenames = ['github_profile.jpg', 'gi.jpg']
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images)