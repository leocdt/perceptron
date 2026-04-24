import numpy as np
import matplotlib.pyplot as plt



def afficherPoints(D):
    colors = ["red" if o == 0 else "green" for o in D[:,3]]
    row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
    ax.scatter(row_x,row_y,row_z,c = colors)

def afficherPlan(W,b):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)            
    ax.set_zlim(-5, 5)
    colors = ["red" if o == 0 else "green" for o in D[:,3]]
    row_x, row_y, row_z = D[:,0], D[:,1], D[:,2]
    ax.scatter(row_x,row_y,row_z,c = colors)
    mesh_range = np.arange(-1.2,1.2,0.1)
    mesh_x,mesh_y = np.meshgrid(mesh_range,mesh_range)
    z = -1 / W[2] * (W[0]*mesh_x + W[1]*mesh_y + b)
    ax.plot_surface(mesh_x,mesh_y,z,alpha=0.4)
    plt.pause(0.5)
    
def calculErreur(D, W, b):
    erreurs = 0
    for i in range(len(D)):
        x, y, z, y_d = D[i]
        y = neurone(x,y,z,W[0],W[1],W[2],b)
        if y != y_d:
            erreurs += 1
    print("Erreur absolue :", erreurs)
#     return erreurs

def neurone(x1,x2,x3,w1,w2,w3,b):
    s = x1*w1 + x2*w2 + x3*w3 + b
    
    if s >= 0:
        y = 1
    else:
        y=0
    return y

def entrainement(D,nb_iter):
    w = np.random.rand(3)
    b = np.random.rand()
    
    
    a = 0.1
    for i in range(nb_iter): 

        numLigne = np.random.randint(0, len(D))
        ligne = D[numLigne]

        # Construire le vecteur d'entrées X et la sortie désirée y_d à partir de la ligne choisie
        vecteurX1 = ligne[0]

        vecteurX2 = ligne[1]

        vecteurX3 = ligne[2]

        vecteurEntrees = np.array([vecteurX1, vecteurX2, vecteurX3])

        yBon = ligne[3]
        # Calculer la sortie prédite y
        y = neurone(vecteurX1, vecteurX2, vecteurX3, w[0], w[1], w[2], b)
        
        # Mettre à jour les poids
        w = w + (a*yBon - a*y) * vecteurEntrees
        # Mettre à jour le biais
        b = b + a*yBon - a*y
        
        if(i%10000 == 0):
            afficherPlan(w,b)
            calculErreur(D,w,b)
    return w, b

def utiliserNeurone(x,y,z, w1, w2, w3, b):
    result = neurone(x,y,z, w1, w2, w3, b)
    if result >= 1:
        print("Green")
    else:
        print("Red")

D = np.loadtxt("donnees.csv",delimiter=",")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
afficherPoints(D)

poidsEtB = entrainement(D, 100000)
poidsFinaux = poidsEtB[0]
bFinal = poidsEtB[1]

x = -0.8831
y = -0.0868
z = -1.9051
utiliserNeurone(x, y, z, poidsFinaux[0], poidsFinaux[1], poidsFinaux[2], bFinal)


