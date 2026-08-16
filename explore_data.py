import numpy as np
import matplotlib.pyplot as plt

# Path root se relative hai — explore_data.py root me hai isliye ye path chalega
class1 = np.loadtxt('Group19/Group19/Classification/LS_Group19/Class1.txt')
class2 = np.loadtxt('Group19/Group19/Classification/LS_Group19/Class2.txt')
class3 = np.loadtxt('Group19/Group19/Classification/LS_Group19/Class3.txt')

print("Class1 shape:", class1.shape)
print("Class2 shape:", class2.shape)
print("Class3 shape:", class3.shape)

plt.scatter(class1[:,0], class1[:,1], label='Class 1', s=10)
plt.scatter(class2[:,0], class2[:,1], label='Class 2', s=10)
plt.scatter(class3[:,0], class3[:,1], label='Class 3', s=10)
plt.legend()
plt.title('LS Data')
plt.savefig('ls_check.png')
print("Saved ls_check.png")

# ---------- NLS data ----------
nls_data = np.loadtxt('Group19/Group19/Classification/NLS_Group19.txt', skiprows=1)
print("NLS total shape:", nls_data.shape)

nls_class1 = nls_data[0:300]
nls_class2 = nls_data[300:800]
nls_class3 = nls_data[800:1800]

plt.figure()  # naya figure, warna purane plot ke upar chadh jayega
plt.scatter(nls_class1[:,0], nls_class1[:,1], label='Class 1', s=10)
plt.scatter(nls_class2[:,0], nls_class2[:,1], label='Class 2', s=10)
plt.scatter(nls_class3[:,0], nls_class3[:,1], label='Class 3', s=10)
plt.legend()
plt.title('NLS Data')
plt.savefig('nls_check.png')
print("Saved nls_check.png")

# ---------- Univariate regression data ----------
uni_data = np.loadtxt('Group19/Group19/Regression/UnivariateData/19.csv', delimiter=',')
print("Univariate shape:", uni_data.shape)

plt.figure()
plt.scatter(uni_data[:,0], uni_data[:,1], s=5)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Univariate Regression Data')
plt.savefig('uni_check.png')
print("Saved uni_check.png")

# ---------- Bivariate regression data ----------
bi_data = np.loadtxt('Group19/Group19/Regression/BivariateData/19.csv', delimiter=',')
print("Bivariate shape:", bi_data.shape)
print("First 3 rows:\n", bi_data[:3])