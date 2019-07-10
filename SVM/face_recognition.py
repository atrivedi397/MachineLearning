import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC    # support vector classifier
from sklearn.model_selection import train_test_split
faces = fetch_olivetti_faces()
print(faces.DESCR)     # description
print(faces.keys())
print(faces.data.shape)
print(faces.images.shape)
print(faces.target.shape)


# helper function to print some images
def print_faces(images, target, top_n):
    # setting up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the image in a matrix of 20x20
        p = fig.add_subplot(20, 20, i+1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the target with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))
    plt.show()


print_faces(faces.images, faces.target, 20)

svc_1 = SVC(kernel='linear')  # by default SVC uses RBF (radial basis function) kernel, we are using linear here

X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=0)
